FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Overridden per-service by the `command:` in docker-compose.yaml
CMD ["python", "blog_scraper.py"]