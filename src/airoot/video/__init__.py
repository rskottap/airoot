from .bin import main
from .lib import *
to_text = video_to_text
and_text_to_text = video_and_text_to_text

def is_video(bytes):
    VIDEO_TYPES = {'mp4', 'webm', 'mkv'}
    from kern import infer_type
    return infer_type(bytes) in VIDEO_TYPES
