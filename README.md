
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

Get started with just a few commands:

1. **Build the Docker image:**
   ```bash
   docker build -t docker_agent .
   ```

2. **Start the Docker container:**
   ```bash
   docker run -p 8000:8000 -it --rm --name docker_agent docker_agent
   ```

   - Binds port `8000` on the host to port `8000` on the container.
   - Removes the container when it stops.
   - Uses the `deepseek-ai/deepseek-coder-1.3b-base` model by default.

3. **Access the API:**
   Once the container is running, you can access the API documentation and test the endpoints by opening a web browser and navigating to:
   ```
   http://localhost:8000/docs
   ```
   This will open the Swagger UI, where you can explore and interact with the available API endpoints.


---

## 🔥 **Enable GPU Acceleration**

Unlock GPU acceleration by adding the `--gpus all` flag:

```bash
docker run -p 8000:8000 --gpus all -it --rm --name docker_agent docker_agent
```

---

## 🔧 **Configuration and Parameter Customization**

The model's behavior and training parameters can be customized by modifying the `src/conf/config.yaml` file. Key configuration options include:

### Model Configuration
- `model_name`: Set the model to use (default: deepseek-ai/deepseek-coder-1.3b-base)
- `context_length`: Set the context length for the model (default: 512)
- `device`: Choose the device to run the model on (default: cpu)
- `use_flash_attention`: Enable or disable flash attention (default: False)

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

## 🤝 **Contributing**

We welcome contributions! Please check out our [Contributing Guide](CONTRIBUTING.md) for details.

---

## 📝 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

