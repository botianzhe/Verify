FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY verify ./verify
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps .
ENV HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
ENTRYPOINT ["verify"]
