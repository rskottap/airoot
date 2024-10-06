#!/usr/bin/env python

__all__ = [
    'video_to_text',
    'video_and_text_to_text',
]

"""
    Warning: the best video-to-text model currently available
    on huggingface is not very good, and only spits out one
    of 400 hard-coded classes. Would love to add a generative
    model that actually describes the video, ideally with less
    hacky video parsing too, but we're not currently aware of
    any models that support this.
"""

import os
import sys


def get_classes():
    # model predicts one of the 400 Kinetics-400 classes
    # kinetics-400 classes can be found here
    url = 'https://gist.githubusercontent.com/willprice/f19da185c9c5f32847134b87c1960769/raw/9dc94028ecced572f302225c49fcdee2f3d748d8/kinetics_400_labels.csv'
    basename = os.path.basename(url)
    cache_path = os.path.expanduser(f'~/.cache/kern/video/{basename}')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        import requests
        content = requests.get(url).content.decode()
        with open(cache_path, 'w') as fp:
            fp.write(content)
    with open(cache_path, 'r') as fp:
        content = fp.read()
    return dict(line.split(',') for line in content.splitlines())


def read_video(container, indices, format="rgb24"):
    """
    Decode the video with PyAV decoder.

    container: av.container.input.InputContainer.
    indices: list[int]. List of frame indices to decode.
    returns: numpy array of decoded frames shape (num_frames, height, width, 3).
    """
    # https://pyav.org/docs/develop/cookbook/basics.html

    import numpy as np
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format=format) for x in frames])


def sample_frame_indices(num_frames_to_sample, video_total_frames):
    #converted_len = int(clip_len * frame_sample_rate)
    #end_idx = np.random.randint(converted_len, seg_len)
    #start_idx = end_idx - converted_len
    end_idx = video_total_frames
    start_idx = 0
    import numpy as np
    indices = np.linspace(start_idx, end_idx, num=num_frames_to_sample)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def video_to_text(video):

    import assure
    from mmry import Cache

    bytes = assure.bytes(video)

    cache = Cache('video_to_text')
    if cache.have_blob(bytes):
        return cache.load_blob(bytes).decode()

    text = video_to_text_nocache(bytes)
    cache.save_blob(bytes, text.encode())
    return text


def video_to_text_nocache(video_bytes):

    import av
    import io
    import kern
    import torch
    from transformers import VivitImageProcessor, VivitForVideoClassification

    format = kern.infer_type(video_bytes)
    container = av.open(io.BytesIO(video_bytes), format=format)

    indices = sample_frame_indices(
        num_frames_to_sample=32,
        video_total_frames=container.streams.video[0].frames,
    )

    video = read_video(container=container, indices=indices, format="rgb24")

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    image_processor = VivitImageProcessor.from_pretrained(
        "google/vivit-b-16x2-kinetics400"
    )
    model = VivitForVideoClassification.from_pretrained(
        "google/vivit-b-16x2-kinetics400"
    )

    inputs = image_processor(list(video), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    classes = get_classes()

    predicted_label = logits.argmax(-1).item()
    return classes[str(predicted_label)]


def video_and_text_to_text(query, video_bytes):
    raise NotImplementedError(
        f"Hiya. The authors of this library are not yet "
        f"aware of any Video Question Answering models. "
        f"If you know of any, please open an issue here "
        f"and send a link so we can add support for it: "
        f"https://github.com/notarealdeveloper/kern/issues"
    )

