AI Song/Cover/Audio Generation using RVC Models

**RVC: Retrieval based Voice Cloning**

## Download existing RVC models

You can search for RVC models trained on existing singers, characters, celebrities, people etc., here:

- ⭐ [AI HUB Discord](https://discord.com/channels/1159260121998827560/1175430844685484042)
- [RVC models website](https://rvc-models.com/)

The downloaded zip file should contain the .pth model file and an optional .index file.

---

## Voice Cloning/Conversion

See ⭐ [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen) and follow instructions there.

For Google Colab GPU usage, see notebook [here](https://colab.research.google.com/github/SociallyIneptWeeb/AICoverGen/blob/main/AICoverGen_colab.ipynb).

For running locally, follow instructions in repo.

In a **separate python virtual environment**, do:
```bash
git clone https://github.com/SociallyIneptWeeb/AICoverGen
cd AICoverGen
pip install -r requirements.txt

python src/download_models.py
```

Run the **WebUI**:

```bash
python src/webui.py
```

- Upload your downloaded RVC models in the `Upload model` tab.
- Generate AI Audio in the `Generate` tab with desired voice convertion and audio mixing options.

**Extra:** Full guide (for different tool) here: [AI covers, RVC with crepe](https://youtu.be/bP8AMf20MAY)

### Troubleshooting

For Debian/Ubuntu-based systems:

```bash
sudo apt update
sudo apt install python3-dev
sudo apt install build-essential
```

---

### Basic Terminology and Settings

TODO

---

## Training Your Own RVC Model
Notes taken from [here](https://docs.google.com/document/d/13ebnzmeEBc6uzYCMt-QVFQk-whVrK4zw8k7_Lw3Bv_A/edit?pli=1&tab=t.0#heading=h.bjzhhhcn3f69)

---

## Dataset Creation

Get the highest quality possible audio sources (.flac preferred over mp3s or YouTube rips, but lower quality stuff will still function). 
Ideally you have actual official acapellas, but those are extremely hard to come by for most music.

### Audio/Vocals Seperation, Isolating instrumentals/noise

[UltimateVocalRemover](https://github.com/Anjok07/ultimatevocalremovergui/releases)
- Small UVR video explanation: https://youtu.be/ITNeuOarHHw

[Music Seperation Repo](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model)

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

---

## Training
TODO