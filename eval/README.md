# Evaluation

This directory contains scripts for evaluating how the finetune endpoint affects performance on a project and general coding benchmarks. Note that the eval was a WIP when this project was archived. The basic eval script is functional, but the RAG eval script is not feature complete.

## Running the Evaluation

To run an eval, you first need to add your own data as described in the [Evaluation Algorithm](#evaluation-algorithm) section below. The data should mimic the types of projects you might want to finetune and get suggestions for.

You can run the evaluation by navigating to the `eval` directory and running the following command:

```bash
python run_eval.py
```

You can run eval RAG specific performance by running the following command:

```bash
python run_rag_eval.py
```

The purpose of the RAG eval is to test how much RAG improves performance on a project. Note that RAG with generic pretrained models did **not** help in our experiments, and the feature implementation was incomplete. We do not recommend using RAG unless you plan to build on the existing implementation.

## Objectives of the Evaluation

- Test how much finetuning on relevant materials improves performance on a project
- Test how much finetuning on irrelevant projects hurts performance on a project
- Test how much finetuning on arbitrary materials hurts performance on general coding benchmarks

## Evaluation Algorithm

1. Assume data in `data/` with the following directory structure:

    ```
    metadata.json
    data/{project_name}/
        train/  # This is the code that will be fine-tuned on
            code/
                {code_file_and_dirs}
            documents/
                {doc_file_and_dirs} (currently unused, in the future could support documents like PDFs)
            links.txt (each link should be separated by a newline)
        test/  # This is the code for which we will generate suggestions
            {test_code_file_and_dirs}
    ```

    Our official eval dataset can be downloaded separetely from the repository at [this](https://drive.google.com/file/d/1Z6wagLBtkknVKnjQ4mWQLK5UCgE6VGNc/view?usp=sharing) Google Drive link.

2. Evaluate on generic coding benchmarks with a fresh model.

3. Load each of the project directories and corresponding test directories.

4. For each project:
    - Reinitialize a new model.
    - Evaluate on the test data.
    - Train on the training data with the finetune endpoint.
    - Evaluate on the test data again.
    - If specified in `metadata.json`, evaluate on generic coding benchmarks.

5. When evaluating, log the following metrics:
    - Loss on training data
    - Loss on test data
    - Change in loss on training data
    - Change in loss on test data
    - Change in loss on generic coding benchmarks

6. After evaluating on all projects, collate the results to log:
    - Averages of the above metrics
    - Change in test loss averaged by project type (related, unrelated)

## Metadata

Each project directory should contain a `metadata.json` file with the following fields:
- `test_type`: one of `related`, `unrelated`
- `generic_benchmarks`: a list of generic benchmarks to evaluate on, valid values: {`HumanEval`, `all`}