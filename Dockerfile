FROM ubuntu:latest

RUN apt update
RUN apt install python3.10 python3-pip python-is-python3 -y

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the transformers library
RUN pip install -r requirements.txt

# Copy the download_models.py script into the container
COPY download_models.py .

# Run the download_models.py script to download the models
RUN python download_models.py

# Copy the rest of the code into the container
COPY . .

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]