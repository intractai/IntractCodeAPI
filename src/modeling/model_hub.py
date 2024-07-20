from src.modeling.model_provider import ModelProvider


def get_model(username: str):
    model_provider = ModelProvider.get_instance()
    return model_provider.get_model(username)


def get_model_utils(username: str):
    model_provider = ModelProvider.get_instance()
    return model_provider.get_model_utils(username)


def get_tokenizer(username: str):
    model_utils = get_model_utils(username)
    return model_utils['tokenizer']


def get_inference_model(username: str):
    model_provider = ModelProvider.get_instance()
    return model_provider.get_inference_model(username)
