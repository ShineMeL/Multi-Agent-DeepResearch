# syntax=docker/dockerfile:1
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

RUN groupadd --gid 10001 deepresearch \
    && useradd --uid 10001 --gid deepresearch --create-home --shell /usr/sbin/nologin deepresearch \
    && mkdir --parents /var/lib/deepresearch/artifacts \
    && chown --recursive deepresearch:deepresearch /var/lib/deepresearch

# Keep dependency installation cacheable and make lock drift a build failure.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY apps ./apps
COPY src ./src
COPY benchmarks ./benchmarks
COPY models ./models
RUN uv sync --frozen --no-dev \
    && chown --recursive deepresearch:deepresearch /app /opt/venv

USER deepresearch
