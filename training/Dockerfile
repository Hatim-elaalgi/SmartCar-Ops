# Training environment = the official Ultralytics GPU image (PyTorch + CUDA + Ultralytics + OpenCV),
# plus JupyterLab (notebook + kernel server for VS Code), papermill (run the notebook from a terminal),
# ONNX tools (export for the dashboard) and ffmpeg (demo clips).
FROM ultralytics/ultralytics:latest

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir jupyterlab ipywidgets ipykernel papermill onnx onnxslim onnxruntime

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/training
EXPOSE 8888
# Jupyter server used as the notebook kernel (JupyterLab in the browser, or VS Code "Existing Jupyter Server").
# The port is published on 127.0.0.1 only (docker-compose.yml); allow_origin / xsrf settings let VS Code connect.
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/workspace --IdentityProvider.token=${JUPYTER_TOKEN:-smartcar} --ServerApp.allow_origin='*' --ServerApp.disable_check_xsrf=True"]
