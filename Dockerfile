FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Optional: YOLO (ultralytics + CPU torch) for running your trained model inside the container.
# Build with:  docker compose build --build-arg WITH_YOLO=1
ARG WITH_YOLO=0
RUN if [ "$WITH_YOLO" = "1" ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
      pip install ultralytics ; \
    fi

COPY . .

EXPOSE 8050
CMD ["python", "app.py"]
