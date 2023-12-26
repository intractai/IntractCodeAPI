from omegaconf import DictConfig

class ConfigProvider:
    """A singleton class that provides the configuration for the server."""

    _instance = None

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise ValueError("The singleton has not been initialized yet.")
        return cls._instance

    @classmethod
    def initialize(cls, cfg: DictConfig):
        cls._instance = cls(cfg)

    @property
    def cfg(self):
        return self._cfg
    
def get_config():
    config_provider = ConfigProvider.get_instance()
    return config_provider.cfg