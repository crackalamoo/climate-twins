# The web UI's bundle (web/ -> climate_twins/static/), without source maps.
FROM node:24-slim AS web
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY tsconfig.json ./
COPY web web
RUN pnpm build && rm climate_twins/static/*.map

FROM python:3.13-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
# FriBiDi: Pillow's raqm text layout for the preview cards (preview.py) loads it at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libfribidi0 && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md LICENSE ./
COPY climate_twins climate_twins
COPY --from=web /app/climate_twins/static climate_twins/static
RUN uv sync --frozen --no-dev

# The data isn't in the image: mount it at /data.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CLIMATE_TWINS_DATA=/data/processed \
    CLIMATE_TWINS_GEONAMES=/data/geonames \
    CLIMATE_TWINS_BOUNDARIES=/data/naturalearth
EXPOSE 8000
CMD ["climate-twins-serve", "--port", "8000"]
