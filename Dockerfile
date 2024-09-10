FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt update && apt-get upgrade -y
RUN apt install python3.10 python3-pip python-is-python3 -y

# Install git
RUN apt install git -y

# Set the working directory in the container
WORKDIR /src

# Copy the requirements file into the container
COPY requirements.txt .

# Install required librarys
RUN pip install -r requirements.txt

# Install flash attention
RUN MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation

WORKDIR /src
COPY src/ .

# Run the application
ENTRYPOINT ["python", "main.py", "server.host=0.0.0.0", "server.port=8000"]
CMD []
EXPOSE 8000

# Example usage:
# docker run -p 8000:8000 your-image-name model.device=cuda train.train_on_code=True
# This allows passing additional arguments to the Python script when running the container