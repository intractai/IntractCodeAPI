# Evaluation

This directory contains scripts for evaluating how the finetune endpoint affects performance on a project and general coding benchmarks.

## Running the Evaluation

WIP

## Objectives of the Evaluation

- Test how much finetuning on relevant materials improves performance on a project
- Test how much finetuning on irrelevant projects hurts performance on a project
- Test how much finetuning on arbitrary materials hurts performance on general coding benchmarks

## Evaluation Algorithm

1. Assume data in `data/` with the following directory structure:

    ```
    metadata.json
    data/{project_name}/
        train/
            code/
                {code_file_and_dirs}
            documents/
                {doc_file_and_dirs} (currently unused, in the future could support documents like PDFs)
            links.txt (each link should be separated by a newline)
        test/
            {test_code_file_and_dirs}
    ```

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
- `generic_benchmarks`: a list of generic benchmarks to evaluate on, valid values: {`HumanEval`, `MBPP`, `all`}