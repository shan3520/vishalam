# Vishalam agent sandbox: runs model-written code with no network, limited memory/CPU,
# read-only filesystem and an unprivileged user (see vishalam/agent/sandbox.py for run flags).
FROM python:3.11-slim
RUN pip install --no-cache-dir pytest==8.3.3
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
