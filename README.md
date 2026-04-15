# 🎬 Automated Video Localization Pipeline
### English → Vietnamese | Zero-Touch Dubbing System

A fully automated pipeline that ingests English videos, translates them, generates natural Vietnamese AI voiceovers with proper background audio ducking, and produces both horizontal (YouTube) and vertical (TikTok/Shorts) renders — all for under $0.01 per video.

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        n8n Orchestrator                       │
│  Webhook → Vast.ai API → SSH → Telegram → SFTP → Destroy    │
└──────────┬────────────────────────────────┬──────────────────┘
           │                                │
    ┌──────▼──────┐                  ┌──────▼──────┐
    │   JOB A     │   Telegram QA    │   JOB B     │
    │  ~3 mins    │ ──── ✅/❌ ────→ │  ~5 mins    │
    │             │   GPU OFF        │             │
    │ • Download  │                  │ • TTS Gen   │
    │ • Demucs    │                  │ • Audio Mix │
    │ • WhisperX  │                  │ • Subtitle  │
    │ • Gemini    │                  │ • FFmpeg    │
    └─────────────┘                  └─────────────┘
         GPU ON                          GPU ON
        $0.003                          $0.006
```

---

## 💰 Cost Breakdown (Per Video)

| Component | Cost |
|-----------|------|
| Vast.ai GPU (~8 min @ $0.07/hr) | $0.009 |
| Gemini 1.5 Pro API | ~$0.01 |
| Google Cloud TTS (1M free chars/mo) | $0.00 |
| FPT.AI TTS (if used instead) | ~$0.10—0.30 |
| **Total (with Google TTS)** | **~$0.02** |
| **Total (with FPT.AI)** | **~$0.12—0.32** |

---

## 🚀 Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed locally
- [n8n](https://n8n.io/) instance (self-hosted or cloud)
- API keys (see below)

### Step 1: Get API Keys

| Service | Get Key From | Free Tier |
|---------|-------------|-----------|
| Vast.ai | https://cloud.vast.ai/account/ | Pay-per-use (~$0.07/hr) |
| Google Gemini | https://aistudio.google.com/apikey | 60 requests/min free |
| Google Cloud TTS | https://console.cloud.google.com/ | 1M chars/month free |
| FPT.AI (optional) | https://console.fpt.ai/ | Limited free tier |
| Telegram Bot | Message @BotFather on Telegram | Free |

### Step 2: Configure

Edit `config/config.env` with your API keys:

```bash
VAST_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
GOOGLE_TTS_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### Step 3: Build & Push Docker Image

```bash
cd d:\n8n

# Build the image
docker build -t your_dockerhub_user/video-localize:latest -f docker/Dockerfile .

# Push to Docker Hub (so Vast.ai can pull it)
docker push your_dockerhub_user/video-localize:latest
```

> ⚠️ The first build takes ~15 minutes as it downloads WhisperX and Demucs models.
> Subsequent Vast.ai instances pull the pre-built image in ~60 seconds.

### Step 4: Import n8n Workflow

1. Open your n8n instance
2. Go to **Workflows** → **Import from File**
3. Select `n8n_workflows/full_pipeline.json`
4. Configure credentials:
   - **Vast.ai API Key**: Create an "HTTP Header Auth" credential with header `Authorization` and value `Bearer YOUR_VAST_API_KEY`
   - **Vast.ai SSH**: Will be configured dynamically per instance
   - **Telegram Bot**: Create a Telegram API credential with your bot token
5. Set environment variables in n8n Settings:
   - `TELEGRAM_CHAT_ID`
   - `VAST_DOCKER_IMAGE`
6. Activate the workflow

### Step 5: Trigger

Send a POST request to the webhook:

```bash
curl -X POST https://your-n8n-instance.com/webhook/localize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

Or use the Telegram bot to submit URLs (extend the workflow as needed).

---

## 📁 Project Structure

```
d:\n8n\
├── config/
│   └── config.env                 # API keys & pipeline settings
├── docker/
│   └── Dockerfile                 # GPU instance image
├── scripts/
│   ├── pipeline.py                # Master orchestration (entry point)
│   ├── checkpoint.py              # Crash recovery system
│   ├── separator.py               # Demucs audio separation
│   ├── transcriber.py             # WhisperX transcription
│   ├── translator.py              # Gemini translation
│   ├── tts_generator.py           # TTS (FPT.AI + Google Cloud)
│   ├── subtitle_generator.py      # ASS subtitle generation
│   └── renderer.py                # FFmpeg rendering
├── n8n_workflows/
│   └── full_pipeline.json         # Importable n8n workflow
├── assets/
│   ├── avatar_idle.png            # PNG-Tuber idle state
│   └── avatar_speaking.png        # PNG-Tuber speaking state
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🔧 Configuration Reference

### TTS Provider

**Google Cloud TTS** (recommended — better quality, 1M free chars/month):
```env
TTS_PROVIDER=google
GOOGLE_TTS_API_KEY=your_key
GOOGLE_TTS_VOICE=vi-VN-Neural2-A
```

Available Vietnamese voices:
| Voice | Gender | Type |
|-------|--------|------|
| `vi-VN-Neural2-A` | Female | Neural2 (best quality) |
| `vi-VN-Neural2-D` | Male | Neural2 |
| `vi-VN-Standard-A` | Female | Standard |
| `vi-VN-Standard-B` | Male | Standard |
| `vi-VN-Wavenet-A` | Female | WaveNet |
| `vi-VN-Wavenet-B` | Male | WaveNet |
| `vi-VN-Wavenet-C` | Female | WaveNet |
| `vi-VN-Wavenet-D` | Male | WaveNet |

**FPT.AI** (alternative — specialized Vietnamese regional accents):
```env
TTS_PROVIDER=fpt
FPT_API_KEY=your_key
FPT_VOICE=banmai
```

Available FPT.AI voices:
| Voice | Gender | Region |
|-------|--------|--------|
| `banmai` | Female | Northern |
| `thuminh` | Female | Northern |
| `leminh` | Male | Northern |
| `myan` | Female | Central |
| `giahuy` | Male | Central |
| `lannhi` | Female | Southern |
| `linhsan` | Female | Southern |

### Audio Sync Tuning

```env
ATEMPO_MAX=1.15        # Max speed-up before extended bleed (1.0 = no speedup)
BLEED_SECONDS=0.5      # Allowed overlap into next segment (seconds)
BGM_BASE_VOLUME=0.3    # Background music base volume (0.0—1.0)
BGM_DUCK_RATIO=10      # How aggressively BGM ducks (higher = more ducking)
```

### Output Modes

```env
RENDER_MODES=horizontal,vertical    # Comma-separated: horizontal, vertical, or both
```

- **horizontal**: 16:9 for YouTube
- **vertical**: 9:16 for TikTok/YT Shorts (blurred background + centered video)

---

## 🔄 Crash Recovery

The pipeline uses atomic checkpoints via `status.json`. If the Vast.ai instance crashes mid-processing:

1. n8n detects SSH failure
2. n8n spins up a new instance
3. Re-runs the same pipeline command
4. The script reads `status.json` and skips all completed phases
5. Resumes from the exact point of failure

Individual TTS chunks are saved as separate files, so even a crash mid-TTS-generation only reruns the remaining chunks.

---

## 🛠 Local Testing (Without Vast.ai)

You can test the Python scripts locally if you have a CUDA GPU:

```bash
# Install dependencies
pip install -r requirements.txt

# Job A: Download, separate, transcribe, translate
cd scripts
python pipeline.py --mode job_a \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --work-dir ../workspace \
  --config ../config/config.env

# (Review translation.json manually)

# Job B: TTS, render
python pipeline.py --mode job_b \
  --work-dir ../workspace \
  --config ../config/config.env
```

---

## 📋 Troubleshooting

| Issue | Solution |
|-------|---------|
| CUDA out of memory | Reduce `WHISPERX_BATCH_SIZE` to 8 or 4 |
| FPT.AI timeout | Increase max retry in `tts_generator.py` |
| FFmpeg NVENC not found | Fall back: change `h264_nvenc` to `libx264` in `renderer.py` |
| Subtitle diacritics broken | Ensure Be Vietnam Pro font is installed in Docker image |
| Vast.ai no instances available | Pipeline uses fallback GPU list: 4060 Ti → 3090 → 4070 Ti |
| n8n SSH timeout | Increase SSH timeout in workflow node (default: 600s for Job A, 1800s for Job B) |
