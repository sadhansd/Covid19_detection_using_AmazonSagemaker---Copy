FROM python:3.8-slim-buster

RUN apt update -y && apt install -y \
    awscli \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt clean

WORKDIR /app

COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]