import torch
from TTS.api import TTS

xtts = {
    "voice_conversion": "voice_conversion_models/multilingual/vctk/freevc24",
    "voice_cloning": "tts_models/multilingual/multi-dataset/xtts_v2",
}


class XTTSv2:
    # pip install coqui-tts
    # from TTS.api import TTS

    # !WARNING: Only tested on Python >3.9 and <3.12
    # for text to speech

    def __init__(self, type="voice_cloning"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.type = type
        self.name = xtts[type]
        self.load_model()
        # self.sample_rate = self.model.config.audio_encoder.sampling_rate

    def load_model(self):
        self.tts = TTS(self.name, progress_bar=True).to(self.device)

    def generate(
        self,
        source_wav,
        target_wav=None,
        text=None,
        out_language="en",
        file_path="output.wav",
    ):
        if target_wav is None and text:
            # voice cloning
            self.tts.tts_to_file(
                text=text,
                speaker_wav=source_wav,
                language=out_language,
                file_path=file_path,
            )
            return 0
        if text is None and target_wav:
            # voice conversion
            self.tts.voice_conversion_to_file(source_wav, target_wav, file_path)
            return 0
        raise Exception(f"One and only one of target_wav or text needs to be provided.")


# model = XTTSv2()
# model.generate(source_wav='./bill_hicks_ride_audio.wav', text="Hi, this is Bill Hicks ghost speaking. It's just a ride.")
