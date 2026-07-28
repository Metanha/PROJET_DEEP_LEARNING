from dataclasses import dataclass, asdict

from .asr import ASRModel
from .sentiment import SentimentModel
from .preprocessing import preprocess_audio, AudioValidationError, TARGET_SAMPLE_RATE

@dataclass
class PredictionResult:
    transcription: str
    sentiment: str
    confidence: float
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SentimentPipeline:
    def __init__(self):
        # chargement une seule fois, coûteux (ASR + sentiment)
        self.asr_model = ASRModel()
        self.sentiment_model = SentimentModel()

    def predict(self, audio_path: str) -> PredictionResult:
        try:
            waveform = preprocess_audio(audio_path)
        except AudioValidationError as exc:
            return PredictionResult(transcription="", sentiment="neutre", confidence=0.0, error=str(exc))

        try:
            transcription = self.asr_model.transcribe(waveform, sample_rate=TARGET_SAMPLE_RATE)
        except Exception as exc:
            return PredictionResult(transcription="", sentiment="neutre", confidence=0.0, error=f"Erreur ASR : {exc}")

        if not transcription.strip():
            return PredictionResult(transcription="", sentiment="neutre", confidence=0.0, error="Transcription vide.")

        result = self.sentiment_model.predict(transcription)
        return PredictionResult(transcription=transcription, sentiment=result.label, confidence=result.confidence)


from .pipeline import SentimentPipeline
"""
pipeline = SentimentPipeline()
result = pipeline.predict("tests/samples/dialog.pop")
print(result)
"""