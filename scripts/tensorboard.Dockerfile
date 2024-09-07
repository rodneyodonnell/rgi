FROM python:3.12-slim

RUN pip install --no-cache-dir tensorboard tensorflow

WORKDIR /tensorboard

# Make port 6006 available for TensorBoard
EXPOSE 6006

CMD ["tensorboard", "--logdir", "/logs", "--host", "0.0.0.0", "--port", "6006"]