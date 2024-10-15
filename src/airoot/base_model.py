__all__ = ['BaseModel']

import torch

class BaseModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        raise NotImplementedError("load_model() must be implemented in the derived class")

    def generate(self, *args, **kwargs):
        raise NotImplementedError("run_inference() must be implemented in the derived class")

