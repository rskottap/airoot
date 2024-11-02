# airoot

GenAI 

<img src="include/root_gen_translation.png" alt="Gen" width="60%"/> 

<img src="include/love_ai_translation.png" alt="AI" width="60%"/> 

### Suite of generation models for text, image, audio and video

Simple library and CLI like usage to easily run generation models locally. 
Mainly intended for trying out and swapping different models quickly, with cpu/gpu usage and basic command line interface for converting different input modalaties into others.

Popular LLMs usage, pdf_to_text, image_to_text, OCR support, image generation, inpainting, music and audio generation, transcription, video generation and video transcription. 

---

### Installation

```
pip install airoot
```
**If NVIDIA CUDA GPU available, then do**
```
pip install airoot[gpu]
```
Next, install ParlerTTS (text to speech model on cpu):
```
pip install git+https://github.com/huggingface/parler-tts.git
```

Extra, Dev mode:

`pip install airoot[dev]` for dev dependencies

```bash
make develop # in repo root to install in editable mode with dev dependencies
pre-commit install # set up the pre-commit git hook to run isort and black automatically
pre-commit run --all-files # to manually run the pre-commit checks
```

---

### Audio Module 🔊 🗣️ 🎶

#### Text to Audio (Audio generation)

For generating audio (from text), see `audiogen` in [Audiogen Module](./src/airoot/audio/TextToAudio.md)

#### Audio to Text (Transcription/Translation)

For converting audio to text, i.e., transcription and translation, see `audiototext` in [Transcription Module](./src/airoot/audio/AudioToText.md)

---
