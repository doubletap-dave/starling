FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HUB_CACHE=/cache/huggingface \
    HOST=0.0.0.0 \
    PORT=8000

# ffmpeg is required for iOS M4A/AAC (and mp3/etc). librosa+soundfile alone cannot.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version | head -n1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY config ./config
COPY starling ./starling

# Install exactly from uv.lock (--locked fails the build if the lock is
# stale) so image rebuilds under the same tag cannot silently drift
# torch/NeMo versions -- that drift is what broke NeMo checkpoint loads
# when a rebuild resolved torch >= 2.6 under the unchanged 0.1.9 tag.
# --no-cache keeps uv's wheel cache out of the image layer: with it, the
# venv layer balloons by gigabytes, which made layer export take ~37 min
# and the registry blob commit time out on push.
RUN uv python install 3.12 \
    && uv sync --locked --no-cache --no-dev --extra nemo --python 3.12

ENV PATH="/opt/venv/bin:$PATH"
VOLUME ["/cache/huggingface"]
EXPOSE 8000

CMD ["uvicorn", "starling.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
