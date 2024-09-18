FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev ffmpeg libsm6 libxext6 libmagic-dev libpoppler-dev poppler-utils

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY _defaults.py ./
COPY embed.py ./
COPY parse.py ./
COPY rag_schema.py ./
COPY query.py ./

CMD ["python", "parse.py"]