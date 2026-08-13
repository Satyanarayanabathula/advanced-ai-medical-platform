FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first so the production container
# does not pull CUDA/NVIDIA dependencies.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       torch==2.12.1 \
       torchvision==0.27.1 \
       --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
       -r requirements.txt

COPY app ./app
COPY frontend ./frontend
COPY model ./model
COPY results ./results

RUN mkdir -p /app/data /app/results/gradcam_api

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]