# airoot

GenAI 

<img src="include/root_gen_translation.png" alt="Gen" width="60%"/> 

<img src="include/love_ai_translation.png" alt="AI" width="60%"/> 

Suite of generation models for text, image, audio and video. 

Simple library like usage to easily run generation models locally. 
Mainly intended for trying out and swapping different models quickly, with cpu/gpu usage and basic command line interface for quick usage. 

Popular LLMs usage, pdf_to_text, image_to_text, OCR support, image generation, inpainting, music and audio generation, transcription, video generation and video transcription. 

### Installation

```
pip3 install airoot
```
`pip3 install airoot[dev]` for dev dependencies

**Dev mode:**
```
make develop # in repo root to install in editable mode with dev dependencies
pre-commit install # set up the pre-commit git hook to run isort and black automatically
pre-commit run --all-files # to manually run the pre-commit checks
```
