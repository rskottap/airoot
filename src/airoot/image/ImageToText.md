# Image Captioning / VQA (ImageToText) Module
Quickly generate detailed captions/descriptions for images and do visual question answering tasks!

❗ GPU highly recommended for best outputs and faster inferences.

## Usage
### Via CLI
```
imagetotext <sample_image.png>
```
```
imagetotext <sample_image.png> --prompt <prompt/question> --max-length 512
```
❗ Recommended to use `--prompt/-p` (and `--max-length/-l`) options only on **GPU** for VQA and other tasks. CPU models don't do well with (and so, don't use) the additional prompt. 

```
imagetotext <sample_image.png> -o out.txt
cat <sample_image.png> | imagetotext >> out.txt
```
- Optionally write output text to file. If not given, prints to stdout.

- Also supports piping in audio data and piping out the text to/from upstream/downstream processes.

⭐ Can use together with `imagegen` to generate sample images and describe/vqa it back.
```
TODO
```

### Examples
TODO

### Via Library
In Python

```python
from PIL import image
from airoot.base_model import get_models
from airoot.image import ImageToText, Blip2

# See all available default models for cpu/gpu
available_models = get_models("ImageToText")

# Tries to load best model based on cpu/gpu availability
model = ImageToText()
# model = Blip2() # use directly for testing etc.,

# load image
img_path = "<full-path-to-image-file>"
image = Image.open(img_path).convert("RGB")

description = model.generate(image) # Describes image
answer = model.generate(image, text="How many people are there in this image?") # For visual QA
```

---
## Models Used
TODO

---
## Notes 📝

- `"Salesforce/blip-image-captioning-large"` (and sometimes `"Salesforce/blip-vqa-base"`) (CPU models) don't do well at all with the extra user prompt (just outputs the prompt back and not description or answers). So prompt is not used in the code for `"Salesforce/blip-image-captioning-large"`.

- GPU highly recommended for best outputs, detailed decsriptions and other tasks with user prompts.
