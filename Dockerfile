# Dockerfile
FROM python:3.8

# Install dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

# Copy code
COPY . /app

# Run model server
CMD ["python", "model_server.py"]
