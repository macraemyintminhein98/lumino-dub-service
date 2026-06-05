/**
 * Lumino Dub Service — runs on MINIX
 * Polls MongoDB for queued dub jobs, processes them, serves downloads
 *
 * Setup on MINIX:
 *   cd ~/lumino-dub-service
 *   npm install
 *   cp .env.example .env  (fill in values)
 *   pm2 start index.js --name lumino-dub
 *   pm2 save
 */

require('dotenv').config()
const mongoose = require('mongoose')
const { exec }  = require('child_process')
const { promisify } = require('util')
const express  = require('express')
const fs       = require('fs')
const path     = require('path')
const crypto   = require('crypto')

const execAsync = promisify(exec)

// ── Config ──────────────────────────────────────────────────────────────────

const MONGODB_URI        = process.env.MONGODB_URI
const PORT               = process.env.DUB_PORT || 3020
const TMP_DIR            = process.env.TMP_DIR || '/tmp/lumino-dub'
const GROQ_API_KEY       = process.env.GROQ_API_KEY
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY
const GOOGLE_TTS_KEY     = process.env.GOOGLE_TTS_API_KEY
const POLL_INTERVAL_MS   = 15_000  // poll every 15 seconds
const MAX_CONCURRENT     = 2       // max simultaneous jobs

let activeJobs = 0

// ── Mongoose model (mirrors src/models/DubJob.ts) ────────────────────────────

const DubJobSchema = new mongoose.Schema({
  userId: String, sourceType: String, sourceUrl: String,
  sourceFilename: String, sourcePath: String, sourceDurationSec: Number,
  outputLanguage: { type: String, default: 'burmese' },
  voiceId: String, recapLength: String, atsMode: String,
  addLogo: Boolean, logoPath: String, logoPlacement: String,
  logoSize: String, logoOpacity: Number,
  blurEnabled: Boolean, blurX: Number, blurY: Number, blurW: Number, blurH: Number,
  freezeFrames: Boolean,
  status: { type: String, default: 'queued' },
  progress: { type: Number, default: 0 },
  errorMessage: String, transcript: String, translatedScript: String,
  outputPath: String, downloadToken: String,
  creditsEstimated: Number, creditsUsed: Number,
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
})

const DubJob = mongoose.model('DubJob', DubJobSchema)
const User   = mongoose.model('User', new mongoose.Schema({ credits: Number }, { strict: false }))

// ── DB update helper ─────────────────────────────────────────────────────────

async function update(jobId, fields) {
  await DubJob.updateOne({ _id: jobId }, { $set: { ...fields, updatedAt: new Date() } })
}

// ── Language config ───────────────────────────────────────────────────────────

const LANG_CONFIG = {
  burmese:  { code: 'my-MM', prompt: 'You are a professional Myanmar video narrator. Write a natural Burmese (Myanmar script) narration from this English transcript. Output ONLY the Burmese narration text, nothing else.' },
  thai:     { code: 'th-TH', prompt: 'You are a professional Thai video narrator. Write a natural Thai narration from this English transcript. Output ONLY the Thai narration text, nothing else.' },
  filipino: { code: 'fil-PH', prompt: 'You are a professional Filipino video narrator. Write a natural Filipino/Tagalog narration from this English transcript. Output ONLY the Filipino narration text, nothing else.' },
  english:  { code: 'en-US', prompt: 'Rewrite this transcript as a clean, engaging English narration. Remove filler words. Output ONLY the narration text.' },
}

const RECAP_RATIO = { short: 0.3, medium: 0.6, long: 0.9, auto: 0 }

// ── Step 1: Download via yt-dlp ──────────────────────────────────────────────

async function downloadVideo(jobId, url) {
  const dir     = path.join(TMP_DIR, jobId)
  const outTmpl = path.join(dir, 'source.%(ext)s')
  await fs.promises.mkdir(dir, { recursive: true })

  const cmd = `yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" --merge-output-format mp4 --no-playlist --max-filesize 500m -o "${outTmpl}" "${url}"`
  await execAsync(cmd, { timeout: 300_000 })

  const mp4 = path.join(dir, 'source.mp4')
  if (!fs.existsSync(mp4)) throw new Error('Download failed')
  return mp4
}

// ── Step 2: Transcribe ───────────────────────────────────────────────────────

async function transcribe(videoPath, jobId) {
  const mp3 = path.join(TMP_DIR, jobId, 'audio.mp3')
  await execAsync(`ffmpeg -y -i "${videoPath}" -vn -q:a 5 "${mp3}"`)

  const buf      = fs.readFileSync(mp3)
  const blob     = new Blob([buf], { type: 'audio/mpeg' })
  const formData = new FormData()
  formData.append('file', blob, 'audio.mp3')
  formData.append('model', 'whisper-large-v3')
  formData.append('response_format', 'text')
  formData.append('language', 'en')

  const res = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${GROQ_API_KEY}` },
    body: formData,
  })

  try { fs.unlinkSync(mp3) } catch (_) {}
  if (!res.ok) throw new Error(`Groq failed: ${await res.text()}`)
  return (await res.text()).trim()
}

// ── Step 3: Translate ─────────────────────────────────────────────────────────

async function translate(transcript, language, recapLength) {
  const ratio   = RECAP_RATIO[recapLength] || 0.6
  const trimmed = ratio > 0 ? transcript.slice(0, Math.floor(transcript.length * ratio)) : transcript
  const cfg     = LANG_CONFIG[language] || LANG_CONFIG.burmese

  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${OPENROUTER_API_KEY}`, 'Content-Type': 'application/json',
               'HTTP-Referer': 'https://app.luminoaistudiosmm.com', 'X-Title': 'Lumino Dub' },
    body: JSON.stringify({
      model: 'anthropic/claude-3-haiku', max_tokens: 3000,
      messages: [
        { role: 'system', content: cfg.prompt },
        { role: 'user',   content: `Translate/adapt this transcript:\n\n${trimmed}` },
      ],
    }),
  })
  if (!res.ok) throw new Error(`OpenRouter failed: ${await res.text()}`)
  const data = await res.json()
  return data.choices?.[0]?.message?.content?.trim() || ''
}

// ── Step 4: Generate TTS ─────────────────────────────────────────────────────

async function generateTTS(text, voiceId, langCode, jobId) {
  const outPath = path.join(TMP_DIR, jobId, 'tts.mp3')
  const res = await fetch(`https://texttospeech.googleapis.com/v1/text:synthesize?key=${GOOGLE_TTS_KEY}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input: { text },
      voice: { languageCode: langCode, name: voiceId },
      audioConfig: { audioEncoding: 'MP3', speakingRate: 1.0 },
    }),
  })
  if (!res.ok) throw new Error(`Google TTS failed: ${await res.text()}`)
  const data  = await res.json()
  const audio = Buffer.from(data.audioContent, 'base64')
  fs.writeFileSync(outPath, audio)
  return outPath
}

// ── Step 5: FFmpeg render ────────────────────────────────────────────────────

async function renderVideo(videoPath, audioPath, job) {
  const jobId  = job._id.toString()
  const outDir = path.join(TMP_DIR, jobId)
  let   workingVideo = videoPath

  // Apply blur if enabled
  if (job.blurEnabled && job.blurX != null) {
    const blurred = path.join(outDir, 'blurred.mp4')
    const { stdout } = await execAsync(`ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "${videoPath}"`)
    const [w, h] = stdout.trim().split('x').map(Number)
    const bx = Math.round(job.blurX * w), by = Math.round(job.blurY * h)
    const bw = Math.round(job.blurW * w), bh = Math.round(job.blurH * h)
    const filter = `[0:v]crop=${bw}:${bh}:${bx}:${by},boxblur=15:3[b];[0:v][b]overlay=${bx}:${by}[vout]`
    await execAsync(`ffmpeg -y -i "${videoPath}" -filter_complex "${filter}" -map "[vout]" -map 0:a -c:a copy "${blurred}"`, { timeout: 300_000 })
    workingVideo = blurred
  }

  // Apply logo if enabled
  if (job.addLogo && job.logoPath && fs.existsSync(job.logoPath)) {
    const withLogo = path.join(outDir, 'logo.mp4')
    const positions = { 'top-right':'W-w-20:20', 'top-left':'20:20', 'bottom-right':'W-w-20:H-h-20', 'bottom-left':'20:H-h-20' }
    const pos   = positions[job.logoPlacement] || 'W-w-20:20'
    const scale = job.logoSize === 'large' ? 160 : job.logoSize === 'small' ? 80 : 120
    const alpha = (job.logoOpacity / 100).toFixed(2)
    const filter = `[1:v]scale=${scale}:-1,format=rgba,colorchannelmixer=aa=${alpha}[logo];[0:v][logo]overlay=${pos}[vout]`
    await execAsync(`ffmpeg -y -i "${workingVideo}" -i "${job.logoPath}" -filter_complex "${filter}" -map "[vout]" -map 0:a -c:a copy "${withLogo}"`, { timeout: 300_000 })
    workingVideo = withLogo
  }

  // Replace audio
  const outputPath = path.join(outDir, 'output.mp4')
  await execAsync(`ffmpeg -y -i "${workingVideo}" -i "${audioPath}" -map 0:v -map 1:a -c:v copy -shortest "${outputPath}"`, { timeout: 600_000 })
  return outputPath
}

// ── Main pipeline ─────────────────────────────────────────────────────────────

async function processJob(job) {
  const jobId = job._id.toString()
  console.log(`[dub] Processing job ${jobId}`)
  activeJobs++

  try {
    // Get video
    await update(jobId, { status: 'downloading', progress: 5 })
    let videoPath
    if (job.sourceType === 'url') {
      videoPath = await downloadVideo(jobId, job.sourceUrl)
    } else {
      videoPath = job.sourcePath
      if (!fs.existsSync(videoPath)) throw new Error('Uploaded file not found')
    }

    // Get duration
    const { stdout: durOut } = await execAsync(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${videoPath}"`)
    const durationSec = parseFloat(durOut.trim())
    await update(jobId, { sourceDurationSec: durationSec, progress: 15 })

    // Transcribe
    await update(jobId, { status: 'transcribing', progress: 20 })
    const transcript = await transcribe(videoPath, jobId)
    await update(jobId, { transcript, progress: 40 })

    // Translate
    await update(jobId, { status: 'translating', progress: 45 })
    const langCfg = LANG_CONFIG[job.outputLanguage] || LANG_CONFIG.burmese
    const translatedScript = await translate(transcript, job.outputLanguage, job.recapLength)
    await update(jobId, { translatedScript, progress: 60 })

    // TTS
    await update(jobId, { status: 'generating_voice', progress: 65 })
    const audioPath = await generateTTS(translatedScript, job.voiceId, langCfg.code, jobId)
    await update(jobId, { progress: 75 })

    // Render
    await update(jobId, { status: 'rendering', progress: 80 })
    const outputPath = await renderVideo(videoPath, audioPath, job)
    await update(jobId, { progress: 95 })

    // Done — calculate actual credits
    const creditsUsed  = Math.max(10, Math.ceil((durationSec / 60) * 2))
    const overcharge   = (job.creditsEstimated || 0) - creditsUsed
    await update(jobId, { status: 'complete', progress: 100, outputPath, creditsUsed })

    // Refund overcharge
    if (overcharge > 0) {
      await User.updateOne({ _id: job.userId }, { $inc: { credits: overcharge } })
    }

    // Clean up intermediates (keep output.mp4)
    for (const f of ['source.mp4', 'audio.mp3', 'blurred.mp4', 'logo.mp4', 'tts.mp3']) {
      try { fs.unlinkSync(path.join(TMP_DIR, jobId, f)) } catch (_) {}
    }

    console.log(`[dub] ✓ Job ${jobId} complete (${durationSec}s, ${creditsUsed} credits)`)

  } catch (err) {
    console.error(`[dub] ✗ Job ${jobId} failed:`, err.message)
    await update(jobId, { status: 'failed', errorMessage: err.message })
    // Refund all estimated credits on failure
    await User.updateOne({ _id: job.userId }, { $inc: { credits: job.creditsEstimated || 0 } })
    // Clean up all temp files on failure
    try { fs.rmSync(path.join(TMP_DIR, jobId), { recursive: true, force: true }) } catch (_) {}
  } finally {
    activeJobs--
  }
}

// ── Poll loop ─────────────────────────────────────────────────────────────────

async function poll() {
  if (activeJobs >= MAX_CONCURRENT) return

  try {
    const jobs = await DubJob.find({ status: 'queued' })
      .sort({ createdAt: 1 })
      .limit(MAX_CONCURRENT - activeJobs)

    for (const job of jobs) {
      // Mark as claimed immediately to avoid double-processing
      const result = await DubJob.findOneAndUpdate(
        { _id: job._id, status: 'queued' },
        { $set: { status: 'downloading', updatedAt: new Date() } }
      )
      if (result) processJob(job).catch(console.error)
    }
  } catch (err) {
    console.error('[dub] Poll error:', err.message)
  }
}

// ── Download server ───────────────────────────────────────────────────────────
// Served via Cloudflare tunnel — Vercel sends download URLs pointing here

const app = express()

app.get('/dub/download/:jobId', async (req, res) => {
  try {
    const { jobId } = req.params
    const { token } = req.query

    const job = await DubJob.findById(jobId).select('outputPath downloadToken sourceFilename status')
    if (!job || job.status !== 'complete') return res.status(404).json({ error: 'Not found' })
    if (job.downloadToken !== token)       return res.status(403).json({ error: 'Invalid token' })
    if (!fs.existsSync(job.outputPath))    return res.status(404).json({ error: 'File not found' })

    const filename = `lumino_${(job.sourceFilename || 'dubbed').replace(/\.[^.]+$/, '')}_dubbed.mp4`
    res.setHeader('Content-Type', 'video/mp4')
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`)
    fs.createReadStream(job.outputPath).pipe(res)

    // Clean up after download
    res.on('finish', () => {
      setTimeout(() => {
        try { fs.rmSync(path.dirname(job.outputPath), { recursive: true, force: true }) } catch (_) {}
      }, 5000)
    })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

app.get('/health', (_, res) => res.json({ status: 'ok', activeJobs, uptime: process.uptime() }))

// ── Start ─────────────────────────────────────────────────────────────────────

async function start() {
  await fs.promises.mkdir(TMP_DIR, { recursive: true })
  await mongoose.connect(MONGODB_URI)
  console.log('[dub] Connected to MongoDB')

  app.listen(PORT, () => console.log(`[dub] Download server on :${PORT}`))

  setInterval(poll, POLL_INTERVAL_MS)
  poll() // immediate first poll
  console.log(`[dub] Polling every ${POLL_INTERVAL_MS / 1000}s, max ${MAX_CONCURRENT} concurrent jobs`)
}

start().catch(err => { console.error('[dub] Fatal:', err); process.exit(1) })
