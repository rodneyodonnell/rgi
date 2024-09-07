FROM tensorflow/tensorflow:2.17.0-gpu
# FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src src

CMD ["python", "src/main.py"]
