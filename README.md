# Docker Agent

This Docker image is based on the Python 3 slim image and includes the transformers and dioc libraries.

## Building the Docker Image

To build the Docker image, navigate to the directory containing the Dockerfile and run the following command:

```bash
docker build -t docker_agent .
```

## Starting the Docker Container

To start the Docker container, run the following command:

```bash
docker run -p 8000:8000 -it --rm --name docker_agent docker_agent
```

## Testing the Docker Container

To test the Docker container, run the following command:

```bash
python call_agent.py "<INPUT TEXT>"
``` 