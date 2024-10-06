from .image import *
from .image_and_text import *

def is_image(bytes):
    IMAGE_TYPES = {'jpg', 'png', 'gif', 'bmp', 'webp'}
    from kern import infer_type
    return infer_type(bytes) in IMAGE_TYPES
