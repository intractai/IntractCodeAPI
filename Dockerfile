FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt update
RUN apt install python3.10 python3-pip python-is-python3 -y

# Set the working directory in the container
WORKDIR /src

# Copy the requirements file into the container
COPY requirements.txt .

# Install required librarys
RUN pip install -r requirements.txt
RUN apt install git -y

# Install flash attention
RUN git clone --branch v2.3.6 --depth 1 https://github.com/Dao-AILab/flash-attention.git
WORKDIR /src/flash-attention
RUN MAX_JOBS=4 python setup.py install

WORKDIR /src
COPY src/ .

# Run the application
# CMD ["ls"]
CMD ["python",  "main.py", "server.host=0.0.0.0", "server.port=8000"]
EXPOSE 8000