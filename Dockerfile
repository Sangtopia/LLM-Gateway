FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway ./gateway

EXPOSE 8000
# One worker process: the circuit breaker and latency tracker are per process, which is
# fine, but run more containers rather than more workers to scale out.
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
