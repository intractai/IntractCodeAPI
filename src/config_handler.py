from omegaconf import DictConfig


class ConfigProvider:
    """A singleton class that provides the configuration for the server."""

    _instance = None

    def __init__(self, config: DictConfig):
        self._config = config

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise ValueError("The singleton has not been initialized yet.")
        return cls._instance

    @classmethod
    def initialize(cls, config: DictConfig):
        cls._instance = cls(config)

    @property
    def config(self):
        return self._config
    
    
def get_config():
    config_provider = ConfigProvider.get_instance()
    return config_provider.config