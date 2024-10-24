# TextToAudio Module

❗GPU highly recommended for good quality and inference times.

## Usage
### Via CLI
```
audiogen <example text> --type <speech/music> --output example.wav
```
```
audiogen "Hello, how are you?" --type speech --output hello.wav
```
- Supports piping in text and piping out audio array to/from other commands.

`echo "Hello, how are you?" | audiogen --type speech`

❗Only outputs `.wav` file for best quality. Use `ffmpeg` to convert from wav to other audio formats.

#### Speech
```
audiogen "Hello, nice to meet you"
```
`type` defaults to speech. Writes to `output.wav` by default.

❗Example **voice preset for speech models only**. 
Uses Bark model on GPU, defaults to `v2/en_speaker_6`.
```
audiogen "Hello, nice to meet you" --type speech --voice-preset v2/en_speaker_9 --output hello2.wav
```
Example voice preset for Parler-TTS on CPU:
```
audiogen "Hello, nice to meet you" --type speech --voice-preset Lea --output hello2.wav
```

#### Music
```
audiogen "90s rock song with loud guitars and heavy drums" --type music --output rock.wav
```
❗**For music only**, optionally provide `-n/--len` option for length of generated audio in *seconds*. Recommended to generate shorter clips ~5 sec on CPU for reasonable inference times.
```
audiogen "The sound of a hammer hitting a wooden surface." --type music --len 10 --output hammer.wav
```

Do `audiogen --help` for full CLI usage. 

**See *Models Used* section for model links, available voice presets and defaults for CPU vs GPU.**

---

### Via Library
```python
from airoot.base_model import get_models
from airoot.audio import TextToAudio, Bark
import soundfile as sf

# Default models to use for the TextToAudio module based on CPU/GPU if they can be loaded successfully.
default_models = get_models("TextToAudio")

# Loads default speech model class based on machine
model = TextToAudio('speech') # or 'music' for music models
print(type(model))

audio_array = model.generate("Hello, nice to meet you!")
sf.write("hello.wav", audio_array, model.sample_rate)

# Can directly use model, instead of default. Recommended only for testing/dev.
model = Bark("suno/bark-small")
audio_array = model.generate("Hello, nice to meet you!", voice_preset="v2/en_speaker_4")
```

**See *Notes* section below for more detail**s.
See `text_to_audio.py` for all available models and implementation details. 

---

## Models Used

|         | Text To Speech                                      | Text To Music                                       |
|---------|---------------------------------------------|---------------------------------------------|
| **GPU** | Suno Bark &#124; [HF Link](https://huggingface.co/docs/transformers/main/en/model_doc/bark) &#124; [GitHub Repo](https://github.com/suno-ai/bark)        | AudioCraft - MusicGen &#124; [HF Link](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen_melody#text-only-conditional-generation) &#124; [GitHub Repo](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)        |
| **CPU** | Parler-TTS &#124; [GitHub Repo](https://github.com/huggingface/parler-tts)        | MusicGen Small &#124; [HF Link](https://huggingface.co/facebook/musicgen-small) &#124; [GitHub Repo](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)       |

---

### Bark 
For text to speech on GPU.

**Speaker Library for Bark voice presets:**
https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c

⚠️ `--voice-preset` for Bark needs to be one of these. Defaults to the recommended `v2/en_speaker_6`.

Use cues like `[laughs]`, `[sighs]`, `[gasps]` or `...` for hesitations, in the text prompt for these extra non-speech sounds/effects. 

See the GitHub repo link above for Bark for more details on voice presets, sounds, languages supported etc.,

---

### AudioCraft MusicGen Models 
For text to music on CPU and GPU. 

Good for generating music with prompts like "80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums" etc., 

Uses MusicGen *small* model for CPU. **Takes ~13 seconds for model to load and ~78 seconds to generate a 5 second clip on 4 core cpu.**

⚠️ All audiocraft **medium sized** models NEED a gpu to run. **Recommended >16 GB of GPU Memory for these.**

Extras (for reference):

**AudioGen:**
- Github: https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md

**Magnet:**
- GitHub: https://github.com/facebookresearch/audiocraft/blob/main/docs/MAGNET.md

---

### Parler-TTS
For text to speech on CPU.

`--voice-preset` can be a random voice (for ex: "A female voice") or a specific voice from the available list of speakers (for ex: "Jon", "Lea").

See [here](https://github.com/huggingface/parler-tts?tab=readme-ov-file#-using-a-specific-speaker) for full list of available speakers. 

---

## Extras

|         | Text To Speech                                      |
|---------|---------------------------------------------|
| **XTTS** | [HF Link](https://huggingface.co/coqui/XTTS-v2) &#124; [GitHub Repo](https://github.com/coqui-ai/TTS)        |
| **MeloTTS** | [HF Link](https://huggingface.co/myshell-ai/MeloTTS-English) &#124; [GitHub Repo](https://github.com/myshell-ai/MeloTTS)        |

|         | Text To Music                                      |
|---------|---------------------------------------------|
| **StableAudio 1.0** | [HF Link](https://huggingface.co/stabilityai/stable-audio-open-1.0) &#124; [HF Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_audio)       |

### XTTS and MeloTTS
XTTS (has voice cloning) and MeloTTS (real-time cpu inference) for text to speech.

BLOG: https://bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models

Both supposedly very good but had all sorts of installation errors with fixed python versions and pip package installations.
Not suported on all OS'es either.
So using native HuggingFace developed Parler-TTS instead. Less features, but easy integration with HF and the general structure used here.

---

### StableAudio1
For text to music. Very slow on CPU.

⚠️ **Need to log in to HuggingFace for access.** 
Do `huggingface-cli login` and create/paste in an access token from your Account->Settings->Access Tokens. 

---

## Notes

- Do `get_models("TextToAudio")` in python to see the config for default models to use for speech and music based on CPU vs GPU availablity. 

- The **first time** the audiogen command is run (for speech or music), the script `test_load_model.py` is run with the different models available (in order), and sets the **first model that can be successfully loaded into memory** as the **default model** for that machine. Writes the default model config to `~/.cache/airoot/audio/model.json`. 
    - This can take long the first time, so please allow it some time.
    - For subsequent runs, by default uses the model in this file (if file exists) and doesn't try to re-check again. So, next runs should be slightly faster.
    - To force re-checking compatability again, simply remove the file `rm ~/.cache/airoot/audio/model.json`. 
