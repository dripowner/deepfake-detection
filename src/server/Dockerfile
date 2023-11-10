FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /server

COPY requirements.txt .

RUN apt update -y && apt install -y python3.10 python3-pip

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install facenet-pytorch==2.5.3

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "main.py"]