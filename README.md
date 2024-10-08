
# Intract Code API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

An API designed for code completion and fine-tuning of open-source large language models on internal codebases and documents. 

## ✨ **Key Features**

- 🚀 **Code Completion API**: Seamlessly integrate advanced code suggestions into your development process.
- ⚙️ **Custom Fine-tuning**: Personalize models to your company's codebase and internal knowledge, including support for documents and PDFs.
- 📈 **Fine-tuning Techniques**: Supports Standard, LoRA, and QLoRA fine-tuning methods.
- 👥 **Multi-user Support**: Run multiple users with different models on a shared server.
- 🧠 **Retrieval-Augmented Generation (RAG)**: Experimental feature enabling context-aware generation.

---

## 🚀 **Quick Start**

We provide instructions for running the API with and without Docker. Follow either the [Without Docker](#-without-docker) or [With Docker](#-with-docker) section, and then follow the instructions in the [Testing the API](#-testing-the-api) section to get started.

### 🖥️ **Without Docker** (Recommended)

1. **Install dependencies:**
   Ensure you have Python 3.9+ and pip installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install CUDA (recommended):**
   If you want to use GPU acceleration, make sure you have CUDA installed. The version of flash attention that we use is only compatible with CUDA 12.2 and 11.8. You can instead build it from [source](https://github.com/Dao-AILab/flash-attention) if you want to use a different version of CUDA, but this takes a lot longer and is more work.

3. **Install Flash Attention (recommended):**
   Installing flash attention will significantly improve performance and is highly recommended:
   ```bash
   MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation
   ```

4. **Set OpenAI API Key (optional):**
   If you want to use SFT with rejection sampling or RAG, you need to set the OPENAI_API_KEY environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

5. **Start the API:**
   Navigate to the `src` directory and run:
   ```bash
   python main.py
   ```

   - This starts the server on `localhost:8000`.
   - Uses the `deepseek-ai/deepseek-coder-1.3b-base` model by default.
   - Add a `--config-name=cpu_config` flag to run on CPU instead of GPU (extremely slow).

### 🐳 **With Docker**

You can also use the provided `Dockerfile` to run the API:

1. **Build the Docker image:**
   ```bash
   docker build -t intract_api .
   ```

2. **Set OpenAI API Key (optional):**
   If you want to use SFT with rejection sampling or RAG, you need to set the OPENAI_API_KEY environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

3. **Start the Docker container:**
   ```bash
   docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY -it --rm --name intract_api intract_api --config-name=cpu_config
   ```

   - Binds port `8000` on the host to port `8000` on the container.
   - Removes the container when it stops.
   - Uses the `deepseek-ai/deepseek-coder-1.3b-base` model by default.

4. **Enable GPU Acceleration (recommended):**
   To use GPU acceleration, add the `--gpus all` flag when starting the container, and remove the `--config-name cpu_config` flag to revert to the default, gpu-compatible config:
   ```bash
   docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY --gpus all -it --rm --name intract_api intract_api
   ```

### 🧪 **Testing the API**

Once the server is running (either with or without Docker), you can test the API:

1. Open a web browser and navigate to `http://localhost:8000/docs` to access the Swagger UI, where you can explore and interact with the available API endpoints.
2. Complete the following steps to set up and authorize your session. This is required for using the other endpoints:
   - Navigate to `localhost:8000/register` to create an account (data will only be stored locally).
   - Return to the Swagger UI at `localhost:8000/docs`.
   - Click the "Authorize" button on the top right of the Swagger UI to authorize your session.
3. You can now test any of the endpoints through the Swagger UI, such as:
   - `/generate` to get a code completion
   - `/finetune/project` to start a fine-tuning process

---

## 🔧 **Configuration and Parameter Customization**

The model's behavior and training parameters can be customized by modifying the `src/conf/config.yaml` file. Key configuration options include:

### Model Configuration
- `model_name`: Set the model to use (default: deepseek-ai/deepseek-coder-1.3b-base)
- `context_length`: Set the context length for the model (default: 512)
- `device`: Choose the device to run the model on (default: cuda)
- `use_flash_attention`: Enable or disable flash attention (default: True)

### Fine-tuning Method Selection
You can switch between different fine-tuning methods by adjusting the following parameters:

#### Standard Fine-tuning
Set `model_type: standard` in the configuration.

#### LoRA (Low-Rank Adaptation)
Set `model_type: lora` and adjust these parameters:
- `lora_r`: Rank of the LoRA update matrices (default: 64)
- `lora_alpha`: LoRA scaling factor (default: 16)
- `lora_dropout`: Dropout probability for LoRA layers (default: 0.01)

#### QLoRA (Quantized LoRA)
Set `model_type: qlora` and adjust these parameters:
- `bits`: Quantization bits (default: 4)
- `double_quant`: Enable double quantization (default: True)
- `quant_type`: Quantization data type (default: nf4)
- `optim`: Optimizer for QLoRA (default: paged_adamw_32bit)
- `gradient_checkpointing`: Enable gradient checkpointing (default: True)

### Training Configuration
- `max_gen_length`: Maximum length of generated code (default: 128)
- `max_revision_steps`: Maximum number of code revision steps (default: 2)
- `use_ntp` and `use_fim`: Enable/disable specific training techniques
- `train_on_code`, `train_on_docs`, etc.: Configure what to train on

For a complete list of configurable parameters, refer to the `src/conf/config.yaml` file in the project repository.

---

## 📄 **Documentation**

Explore the full API documentation by visiting `http://localhost:8000/docs` after starting the server.

---

## 🧠 **How Fine-Tuning Works**

Our fine-tuning process is versatile and powerful, supporting multiple approaches:

### 🔄 Types of Fine-Tuning
- Self-supervised learning
    - Next Token Prediction (NTP)
    - Fill-in-the-Middle (FIM)
- Supervised fine-tuning (SFT) with rejection sampling

### 🛠️ Fine-Tuning Techniques
- Standard fine-tuning
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)

Standard fine-tuning will provide the best results, but it is also the most expensive. LoRA and QLoRA use less memory, but may not be as accurate, and were slower in our experiments.

### 🤖 Self-Supervised Fine-Tuning
We employ two main techniques:
1. **Next Token Prediction (NTP)**: Trains the model to predict the next token in a sequence.
2. **Fill-in-the-Middle (FIM)**: Masks out a portion of the input and trains the model to reconstruct it.

These methods can be applied to various data sources:
- User's codebase
- Documentation text
- Code snippets extracted from documentation
- Auto-generated problems and solutions
- External documents (text files and PDFs)

The fine-tuning process is highly configurable through the config file:
- Choose data sources: `train_on_code`, `train_on_docs`, `train_on_doc_code`, `train_on_practice_problems`, `train_on_verified_solutions`, `train_on_documents`
- Select training methods: `use_ntp`, `use_fim`

### 🎯 SFT with Rejection Sampling
This approach entails generating and solving synthetic problems to improve the model's performance. The full process entails the following steps:

1. Automatically generate problem statements
2. Model produces multiple solutions for each problem
3. Solutions are executed and evaluated automatically
4. This process is repeated for a number of iterations until a solution is found or the maximum number of revisions is reached.
5. The model is trained on the solved problems and their solutions.

Key features:
- Allows iterative improvement without human intervention
- Automatically assesses solution correctness
- Creates a feedback loop for continual refinement

This method leverages the fact that judging solution correctness is often easier than generating correct solutions from scratch, enabling the model to enhance its problem-solving skills over multiple iterations.

---

## 🏗️ **Project Structure**

- `main.py` - Entry point to running the server.
- `modeling.py` - Handles the construction, loading, and management of language models and tokenizers. It includes:
  - A `ModelLoader` class for creating models with various configurations.
  - A `ModelProvider` singleton class that manages model instances for multiple users, allowing retrieval of user-specific models.
  - Utility functions and classes to support model operations and tokenization.
- `config_handler.py` - Contains a singleton class `ConfigProvider` that manages the configuration for the server. It provides methods to initialize the configuration, retrieve the configuration instance, and access the configuration data.
- `database.py` - Manages database operations and connections through a `DatabaseProvider` singleton class, including table creation and connection handling.
- `users.py` - Manages user sessions, authentication, and token handling through a `SessionTracker` singleton and various utility functions. The `SessionTracker` maintains active user sessions, handles user eviction based on inactivity, and manages user-specific resources like models and vector stores.
- `rag.py` - Implements the Retrieval-Augmented Generation (RAG) functionality through a `VectorStoreProvider` singleton class. It manages vector stores for each user, handles document insertion, and provides methods for context retrieval during inference.
- `documents.py` - Handles document processing and conversion, including PDF to text conversion using different libraries (Nougat and PyMuPDF). It also provides caching mechanisms for processed documents and utility functions for handling various document formats.
- `routers/` - Contains the API endpoints for different functionalities:
  - `generator.py` - Handles text generation requests and responses.
  - `fine_tuner.py` - Manages the fine-tuning process for models based on user input.
  - `auth.py` - Handles user authentication, registration, and token management.
- `static/` - Contains static files for authentication and login that are no longer used.
- `training/` - Contains files related to model training and fine-tuning:
  - `data_formatting.py` - Handles data preparation and formatting for training, including functions for tokenization and dataset creation.
  - `finetune.py` - Implements the fine-tuning process, including dataset processing, model configuration, and training loop management.
  - `trainer.py` - Extends the Hugging Face Trainer class to provide custom training functionality. It includes modifications for continual learning, custom evaluation, and memory optimizations.
  - `interactive/` - Contains files for multi-step SFT with rejection sampling. This folder includes implementations for generating and evaluating solutions to programming problems and handling multi-step training processes. It supports features like automated problem generation, solution verification, and iterative improvement of model responses.
- `crawler/` - Contains files for web scraping and document extraction. The crawler functionality uses libraries like Scrapy and BeautifulSoup to extract content from web pages and documentation sites, with explicit support for both GitHub repositories and web-based documentation. It includes utilities for finding documentation URLs and processing HTML content.

---

## 🤝 **Contributing**

> If you want to constribute, we assume that you have read the rest of this document.

We are not longer actively working on this project, and we don't plan to make any updates. If you still want to contribute knowing that, you are welcome to start by making an [Issue](/issues/new) to ask if it is something we would approve. If you do make a pull request, please do your best to follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

---

## 📝 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

