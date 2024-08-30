# Development Guide

The purpose of this document is to help anyone who falls into one of the following categories:
- You want to understand how the project works
- You want to adapt the code base for your own use case
- You want to make a contribution

While we are no longer actively working on this codebase, it is open to anyone to use for any purpose, and we welcome pull requests aimed at improving the base infrastructure we have built. The following sections of this document will arm you with a better understanding of this repository, and how to get started if you want to make changes.

## Table of Contents

- [How it Works](#how-it-works)
    - [Core Features](#core-features)
    - [How Fine-Tuning Works](#how-fine-tuning-works)
    - [Project Structure](#project-structure)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)

## How it Works

### Core Features

- Host an API that provides code completion requests
- Finetuning on text documents, pdfs, and text from a website rooted at a given url
- Standard, LoRA, and QLoRA finetuning
- Multiple users with different models on the same server
- RAG during inference (but does not work well in practice)

### How Fine-Tuning Works

Fine-tuning in this project is a versatile process that supports multiple approaches. The system primarily uses two types of fine-tuning: self-supervised and supervised fine-tuning (SFT) with rejection sampling. Additionally, the system supports different fine-tuning techniques including standard fine-tuning, LoRA (Low-Rank Adaptation), and QLoRA (Quantized LoRA), providing flexibility in the fine-tuning process while managing computational resources effectively.

For self-supervised fine-tuning, the project employs next token prediction (NTP) and fill-in-the-middle (FIM) techniques. NTP trains the model to predict the next token in a sequence, while FIM involves masking out a portion of the input and training the model to reconstruct it. These methods can be applied to various data sources, including the user's codebase, documentation text, and code snippets extracted from documentation. The fine-tuning process is configurable through the config file, allowing users to specify which data sources to use (`train_on_code`, `train_on_docs`, `train_on_doc_code`, etc.) and which training methods to employ (`use_ntp`, `use_fim`).

The project also supports SFT with rejection sampling, particularly for training on practice problems. This approach is fully automated, starting with the generation of problem statements. The model then generates multiple solutions for each problem, which are automatically executed and evaluated. The results are used to filter and select the best solutions for further training. This process, implemented in the `train_multi_step_sft_with_verification` function, allows for iterative improvement of the model's problem-solving capabilities without human intervention. The system automatically assesses the correctness of generated solutions, using valid solutions as targets for supervised fine-tuning. If a solution is incorrect, the model attempts to revise it, creating a feedback loop that continually refines the model's abilities. This approach leverages the fact that it's often easier to judge the correctness of a solution than to generate a correct solution from scratch, allowing the model to improve its problem-solving skills over multiple iterations.

### Project Structure

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

## I Have a Question

> If you want to ask a question, we assume that you have read the rest of this document.

Before you ask a question, it is best to search for existing [Issues](/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue.

If you still have a question, you can open an [Issue](/issues/new), and provide any relevant context. Note that we are not actively working on the repository, and we do not intent to fix most bugs or add aditional features. We are, however, happy to answer questions about the functionality of the project, or help point you in the right direction.


## I Want To Contribute

> If you want to constribute, we assume that you have read the rest of this document.

We are not longer actively working on this project, and we don't plan to make any updates. If you still want to contribute knowing that, you are welcome to start by making an [Issue](/issues/new) to ask if it is something we would approve. If you do make a pull request, please do your best to follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

