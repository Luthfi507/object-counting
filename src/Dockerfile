FROM python:3.12-slim

ENV PATH=/root/.local/bin:${PATH}
ENV PYTHONUNBUFFERED=1

WORKDIR /src

COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    git \
    curl \
    build-essential \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv pip install --system --no-cache-dir -r requirements.txt

EXPOSE 8002

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002"]