import yaml
import datetime
import torch

class Config:
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        data['path'] = path
        #format the timestamp as YYYY-MM-DD_HH:MM:SS
        return Config(data)

    def __init__(self, data):
        self._set_attributes(data)
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        cuda_available = torch.cuda.is_available()
        if not cuda_available and self.device == 'cuda':
            print("WARNING: device set to cuda, but cuda is not available. Using cpu instead.")
            self.device = 'cpu'

    def _set_attributes(self, data, prefix=None):
        for key, value in data.items():
            if isinstance(value, dict):
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._set_attributes(value, new_prefix)
            else:
                attr_name = f"{prefix}.{key}" if prefix else key
                formatted_name = attr_name.replace('.', '_')
                self.__setattr__(formatted_name, value)

    def to_dict(self):
        return self.__dict__
