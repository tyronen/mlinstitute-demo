FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY db.py .
COPY init_db.py .
COPY mnist_model.pth .
COPY models.py .
COPY requirements.txt .
COPY webserver.py .

RUN pip install --no-cache-dir -r requirements.txt
