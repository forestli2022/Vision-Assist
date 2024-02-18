# Stage 1: Build dependencies
FROM python:3.10 AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Create the final production image
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .

EXPOSE 8080

CMD ["gunicorn", "-k", "gevent", "-b", "0.0.0.0:$PORT", "main:app"]
