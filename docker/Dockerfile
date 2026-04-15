# ═══════════════════════════════════════════════════════════
# Video Localization Pipeline — Vast.ai GPU Docker Image
# ═══════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libass-dev \
    git curl wget unzip \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# ── Install Be Vietnam Pro font ──────────────────────────
RUN mkdir -p /usr/share/fonts/truetype/bevietnam && \
    wget -q "https://fonts.google.com/download?family=Be+Vietnam+Pro" -O /tmp/bevietnam.zip && \
    unzip /tmp/bevietnam.zip -d /usr/share/fonts/truetype/bevietnam/ && \
    fc-cache -fv && \
    rm /tmp/bevietnam.zip

# ── SSH server setup (for n8n SSH node) ──────────────────
RUN mkdir /var/run/sshd && \
    echo 'root:vastai_pipeline' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# ── Python dependencies ──────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy pipeline scripts ────────────────────────────────
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY assets/ /app/assets/

# ── Pre-download models on build (saves runtime) ─────────
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')" && \
    python -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8')"

# ── Create workspace directory ───────────────────────────
RUN mkdir -p /workspace

# ── Expose SSH ───────────────────────────────────────────
EXPOSE 22

# ── Start SSH + keep alive ───────────────────────────────
CMD ["/usr/sbin/sshd", "-D"]
