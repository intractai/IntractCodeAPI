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

## Finetune hyperparameters

To change the finetune hyperparameters, the environment variables must start with `FINETUNE_`. For example, to change the number of epochs to 2, add the `-e FINETUNE_NUM_TRAIN_EPOCHS=2` flag to the `docker run` command. The parameters that can be changed are same as the `transformers.TrainingArguments` class, and model arguments for the model being finetuned. For more information, see the [documentation](https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/trainer#transformers.TrainingArguments).

### Finetune on project directories
When an entire project data is passed, the dataset consists of a code completion task for the entire project, and multiple code insertion tasks. The code insertion tasks are randomly generated for each file in the project directory. The hyperparameters controlling the code insertion tasks are as follows:
num_code_insertions_per_file
- `FINETUNE_NUM_CODE_INSERTIONS_PER_FILE`: Number of code insertion tasks per file. Default: 10
- `FINETUNE_SPAN_MAX`: Range of code insertion spans. Default: 256

## Testing the Docker Container

To test the Docker container, run the following command:

```bash
python client/call_agent.py "<INPUT TEXT>"
``` 

To run a sample finetune on project directories, run the following command:

```bash
python client/finetune_agent.py
```

To access the API documentation, navigate to `http://localhost:8000/docs` in your browser.