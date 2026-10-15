FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
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
COPY pyproject.toml README.md ./
COPY config ./config
COPY starling ./starling

RUN uv python install 3.12 \
    && uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv/bin/python -e ".[nemo]"

ENV PATH="/opt/venv/bin:$PATH"
VOLUME ["/cache/huggingface"]
EXPOSE 8000

CMD ["uvicorn", "starling.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
