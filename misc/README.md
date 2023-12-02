Sample command to run supervised training on entire project directories:
```
FINETUNE_USE_CPU=true FINETUNE_NUM_TRAIN_EPOCHS=3 python misc/train_supervised_projectdirs.py microsoft/CodeGPT-small-py
```

Use environment variables to control training specific parameters. The parameters must start with `FINETUNE_`. For example, to change the number of epochs to 2, modify the `FINETUNE_NUM_TRAIN_EPOCHS` environment variable. The parameters that can be changed are same as the `transformers.TrainingArguments` class, and model arguments for the model being finetuned. For more information, see the [documentation](https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/trainer#transformers.TrainingArguments).