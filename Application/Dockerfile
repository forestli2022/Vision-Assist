FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "-k", "gevent", "-b", "0.0.0.0:$PORT", "main:app"]