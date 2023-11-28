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
This will start the container and bind port 8000 on the host to port 8000 on the container. The container will be removed when it is stopped. By default the container will use the `deepseek-ai/deepseek-coder-1.3b-base` model. To use a different model, set the `MODEL_NAME` environment variable when starting the container. For example, to use the `bert-large-uncased` model, add the `-e MODEL_NAME=bert-large-uncased` flag to the `docker run` command. To use GPU acceleration, add the `--gpus all` flag to the `docker run` command.

## Testing the Docker Container

To test the Docker container, run the following command:

```bash
python client/call_agent.py "<INPUT TEXT>"
``` 

To access the API documentation, navigate to `http://localhost:8000/docs` in your browser.