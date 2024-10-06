__all__ = [
    'infer_type',
]

from functools import lru_cache

@lru_cache(maxsize=1)
def infer_type(bytes):
    import filetype
    return filetype.guess_extension(bytes)
