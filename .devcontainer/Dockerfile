FROM mcr.microsoft.com/devcontainers/python:3.10

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/app
COPY requirements.txt /workspaces/app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . /workspaces/app
