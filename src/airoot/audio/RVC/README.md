# AI Song/Cover/Audio Generation using RVC Models

**RVC: Retrieval based Voice Cloning**

## Download existing RVC models

You can search for RVC models trained on existing singers, characters, celebrities, people etc., here:

- ⭐ [AI HUB Discord](https://discord.com/channels/1159260121998827560/1175430844685484042)
- [RVC models website](https://rvc-models.com/)

The downloaded zip file should contain the .pth model file and an optional .index file.

## 🎶 Run Voice Cloning/Conversion

See ⭐ [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen) and follow instructions there.

For Google Colab GPU usage, see notebook [here](https://colab.research.google.com/github/SociallyIneptWeeb/AICoverGen/blob/main/AICoverGen_colab.ipynb).

For running locally, follow instructions in repo.

In a *separate python virtual environment* (**Recommended Python3.10**), do:

Clone the repo:
```bash
git clone https://github.com/SociallyIneptWeeb/AICoverGen
cd AICoverGen
```

Seems like colab notebook has latest requirements.
Install the requirements (taken from colab notebook):
```bash
pip install pip==23.3.1
pip install -r requirements.txt
pip install gradio-client==0.8.1
pip install gradio==3.48.0
# install cuda fix
python -m pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
sudo apt update
sudo apt install sox
```

Download required models:
```bash
python src/download_models.py
```

Run the WebUI:

```bash
python src/webui.py
```

1. Upload your downloaded RVC models in the `Upload model` tab.

2. Generate AI Audio in the `Generate` tab with desired voice conversion and audio mixing options.

**Extra:** Full guide (for different tool) here: [AI covers, RVC with crepe](https://youtu.be/bP8AMf20MAY)

### ❗Troubleshooting

For Debian/Ubuntu-based systems:

```bash
# For missing "Python.h" file
sudo apt update
sudo apt install python3.10-dev # or your python version dev package
sudo apt install build-essential

# Get python3.10
sudo apt install python3.10
sudo apt install libssl-dev libffi-dev python3.10-venv python3.10-distutils python3.10-dev
python3.10 --version

# Make and activate venv
python3.10 -m venv ~/env/rvc
deactivate
source ~/env/rvc/bin/activate
python3 --version

# If venv throws an error that pip is not there, get pip for python3.10
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
```

- `fairseq` doesn't work with python 3.11 but works on 3.10.

- `torch==2.0.1` supports Python versions up to 3.11. **Recommended Python3.10**.

## 📝 Basic Terminology and Settings

1. **Index Rate (RVC Models)**
Definition: Determines the model’s reference for matching vocal characteristics to the target.
Effect on Music: A higher index rate makes the voice sound closer to the source (original) but may reduce diversity in output.

2. **Filter Radius**
Definition: The range over which noise or unwanted artifacts are smoothed out.
Effect on Music: A smaller filter radius keeps more details but may retain noise. A larger radius reduces noise but risks losing fine details.

3. **RMS Mix Rate**
Definition: Adjusts the balance between the root mean square (RMS) power levels of the generated and source audio.
Effect on Music: Controls the loudness and consistency of audio. A higher mix rate makes the output match the input more in energy.

4. **Protect Rate**
Definition: The threshold for preventing certain vocal characteristics from being altered during processing.
Effect on Music: Helps maintain the original voice's character. A high protect rate preserves more of the source vocal traits.

5. **Room Size (Reverb)**
Definition: Represents the simulated size of the room in which the sound is playing.
Effect on Music: Larger room sizes create longer reverb tails, giving a more open and grand sound.

6. **Wetness**
Definition: The level of effect (reverb/echo) applied to the audio.
Effect on Music: Increasing wetness makes the audio sound more distant and spacious.

7. **Dryness**
Definition: The level of the original, unprocessed sound in the mix.
Effect on Music: More dryness makes the sound upfront and clear, with less ambient effect.

8. **Damping Levels**
Definition: Controls the absorption of high frequencies in the simulated space.
Effect on Music: Higher damping makes reverb sound warmer by reducing high frequencies. Lower damping allows more brilliance and sharpness in the reflections.

## Training Your Own RVC Model
Notes taken from [here](https://docs.google.com/document/d/13ebnzmeEBc6uzYCMt-QVFQk-whVrK4zw8k7_Lw3Bv_A/edit?pli=1&tab=t.0#heading=h.bjzhhhcn3f69)

## Dataset Creation

Get the highest quality possible audio sources (.flac preferred over mp3s or YouTube rips, but lower quality stuff will still function). 
Ideally you have actual official acapellas, but those are extremely hard to come by for most music.

### Audio/Vocals Seperation, Isolating instrumentals/noise

[UltimateVocalRemover](https://github.com/Anjok07/ultimatevocalremovergui/releases)
- Small UVR video explanation: https://youtu.be/ITNeuOarHHw

[Music Seperation Repo](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model)
- `pip install demucs onnxruntime` for this.
- For separating all the tracks, sometimes the inference script might look like it's hanging. Check the output folder to see if all the tracks are there and are about the same size and non-zero, if so, you can safely kill the script.

### Removing reverb / echo

It is **necessary** to remove reverb / echo from the song for the best results. Ideally you have as little there as possible in your original song in the first place, and isolating reverb can obviously reduce the quality of the vocal. But if you need to do this, under MDX-Net you can find Reverb HQ, which will export the reverbless audio as the ‘No Other’ option. Oftentimes, this isn’t enough. If that did nothing, (or just didn't do enough), you can try to process the vocal output through the VR Architecture models in UVR to remove the echo and reverb that remains using De-Echo-DeReverb. If that still wasn't enough, somehow, you can use the De-Echo normal model on the output, which is the most aggressive echo removal model of them all.

There’s also a [colab for the VR Arch models](https://colab.research.google.com/drive/16Q44VBJiIrXOgTINztVDVeb0XKhLKHwl?usp=sharing) if you don’t want to run or can’t run UVR locally

### Noise gating to remove silence

Can use [Audacity](https://www.audacityteam.org/download/) for this.
Usually -40db is a good threshold.

- [Adobe Audition](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwjloK3ZkM2JAxXwAK0GHdwtL6UYABAAGgJwdg&ae=2&aspm=1&co=1&ase=2&gclid=Cj0KCQiAire5BhCNARIsAM53K1ipk_EDa-NsGTZa0Pit0AwF9xovG6zm_cFvEU2ep1h3z9Aa5L5uUWAaApBlEALw_wcB&ei=3DguZ6zWPOHs0PEPtqyMqQs&ohost=www.google.com&cid=CAESVOD2Ob4MQvoVlli4Hxngx3MqbS_yYnQ4F2lwVFJb1VvzEZH_2pd4jPH6_yAJd6c59CCdoEpTlK9sw9S_R65wSQfX642qMum-sU5GsBEpeGP6yi5dQQ&sig=AOD64_3MRQl8AC0-csTEKOVgL1Wja7b0vA&q&sqi=2&nis=4&adurl&ved=2ahUKEwistKjZkM2JAxVhNjQIHTYWI7UQ0Qx6BAgKEAE) probably has more advanced tools to do this automatically, but this is a good preset to start off with for people using basic Audacity mixing. 
- If it cuts off mid sentence, redo it with it turned up for the Hold ms.

### Isolating background harmonies / vocal doubling

In most cases, these are too hard to isolate for dataset purposes without it sounding poor quality. The best UVR models for doing so would be 5HP/6HP Karaoke (VR Architecture model) or Karaoke 2 (MDX-Net).

### Audio Length

- The recommendation from the RVC devs is **at least 10 minutes for high quality models** that can handle a variety of pitches and tones. High quality dataset of anywhere between 10 and 45 min should be good.
- Quality of dataset is more important than quantity. Single bigger files under 30 min should be good as is (without breaking up). for longer than 30 min, break into shorter segments. Can use the *regular interval labels feature in Audacity*.
- RVC chops into ~4s bits, so make sure your samples are at least 4s long for consistency reasons (or merge the shorter samples into one long file).

## Training
TODO