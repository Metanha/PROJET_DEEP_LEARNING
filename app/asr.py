from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from .preprocessing import preprocess_audio, TARGET_SAMPLE_RATE

"""
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
print("Chargé.")


waveform = preprocess_audio("tests/samples/SPEAKER_00_pii.wav")
inputs = processor(waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)

with torch.inference_mode():
    logits = model(inputs.input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
print(transcription)
"""

DEFAULT_ASR_MODEL = "jonatasgrosman/wav2vec2-large-english"
#"jonatasgrosman/wav2vec2-large-xlsr-53-french"


class ASRModel:
    def __init__(self, model_name: str = DEFAULT_ASR_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def transcribe(self, waveform, sample_rate: int = 16_000) -> str:
        inputs = self.processor(waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription.strip().lower()
    
"""
asr = ASRModel()
waveform = preprocess_audio("tests/samples/SPEAKER_00_pii.wav")
print(asr.transcribe(waveform, sample_rate=TARGET_SAMPLE_RATE))

"""