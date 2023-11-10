FROM python:3.10.6

WORKDIR /client

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install facenet-pytorch==2.5.3

EXPOSE 8501

COPY . .

CMD ["streamlit", "run", "app.py"]