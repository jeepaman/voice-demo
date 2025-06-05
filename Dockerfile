FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["bash", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
