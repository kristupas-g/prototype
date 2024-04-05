FROM mcr.microsoft.com/devcontainers/miniconda:0-3

RUN conda install -y python=3.8

COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp
WORKDIR /workspaces/prototype

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
