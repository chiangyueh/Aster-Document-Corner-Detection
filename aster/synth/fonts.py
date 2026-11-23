                 
from functools import lru_cache

from PIL import ImageFont


@lru_cache(maxsize=512)
def load(path, size):
    return ImageFont.truetype(path, size)
