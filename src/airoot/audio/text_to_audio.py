### Models
### Bark: https://huggingface.co/docs/transformers/main/en/model_doc/bark, https://github.com/suno-ai/bark

__all__ = [
    'TextToAudio',
    'Bark',
    'MusicGen',
    'get_default_models',
]

import json
import torch
from transformers import AutoProcessor, BarkModel, MusicgenForConditionalGeneration
from airoot.base_model import BaseModel


class MusicGen(BaseModel):

    
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.load_model()
        self.sample_rate = self.model.config.audio_encoder.sampling_rate

    def load_model(self):
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.name, torch_dtype=torch.float16).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.name)

    def generate(self, text, max_new_tokens=256):
        inputs = self.processor(text=[text], padding=True, return_tensors="pt",).to(self.device)
        audio_array = self.model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_new_tokens)
        audio_array = audio_array[0, 0].numpy()
        return audio_array


class Bark(BaseModel):

    #name = "suno/bark" # "suno/bark-small"
    
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.load_model()
        self.sample_rate = self.model.generation_config.sample_rate

    def load_model(self):
        self.model = BarkModel.from_pretrained(self.name, torch_dtype=torch.float16).to(self.device)

        if self.device == 'cuda':
            # if using CUDA device, offload the submodels from GPU to CPU when they’re idle
            self.model.enable_cpu_offload()

        self.model =  self.model.to_bettertransformer() # Better Transformer optimization
        self.processor = AutoProcessor.from_pretrained(self.name)

    def generate(self, text, voice_preset="v2/en_speaker_6"):
        inputs = self.processor(text, voice_preset=voice_preset).to(self.device)
        audio_array = self.model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        return audio_array


# Defaults for speech and music generation in the order they should be tried 
# (if loading fails due to memory costraints)

# Recommended 16GB GPU memory for bark and musicgen-melody 
# bark-small and musicgen-small can be run on smaller GPU memories
# MusicGen models need local GPU. Even bark-small on cpu is too slow.

config = {
    'cpu': {
        'speech': 
                [{'model': Bark, 'name': "suno/bark-small"}], 
        'music': 
                [{'model': Bark, 'name': "suno/bark"}],
        },
    'cuda': {
        'speech': 
                [{'model': Bark, 'name': "suno/bark"}, 
                 {'model': Bark, 'name': "suno/bark-small"}], 
        'music': 
                [{'model': MusicGen, 'name': "facebook/musicgen-melody"}, 
                 {'model': MusicGen, 'name': "facebook/musicgen-small"}],
        },
}

def get_default_models():
    return config

# Default
class TextToAudio(BaseModel):
    def __new__(cls, device, type, *args, **kwds):
        defaults = config[device][type]

        for model in defaults:
            try:
                self = model['model'](name=model['name'], *args, **kwds)
                return self
            except Exception as e:
                ### TODO: del model and memory
                ### TODO: Add gpu vs cpu detection logic here
                continue
        
        raise Exception(f"Unable to load any of the default models for \
                        {type} with {device}:\n \{json.dumps(defaults, indent=4)}")
