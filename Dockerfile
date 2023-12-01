FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt update
RUN apt install python3.10 python3-pip python-is-python3 -y

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the transformers library
RUN pip install -r requirements.txt


# Copy the rest of the code into the container
COPY app/. .
COPY src src

ENV MODEL_NAME=deepseek-ai/deepseek-coder-1.3b-base

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]