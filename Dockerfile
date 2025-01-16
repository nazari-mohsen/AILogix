FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpcre3 \
        libpcre3-dev 

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/ .

EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]