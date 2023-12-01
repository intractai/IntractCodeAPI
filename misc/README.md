Sample command to run supervised training:
```
python misc/train_supervised.py --model_name_or_path microsoft/CodeGPT-small-py --output output/ --data_path misc/test_dataset.json --use_cpu true --num_train_epochs 1 --report_to none
```

Sample command to run supervised training on entire project directories:
```
python misc/train_supervised_projectdirs.py --output_dir='./test' --model_name_or_path=microsoft/CodeGPT-small-py --use_cpu true --model_max_length=1000
```