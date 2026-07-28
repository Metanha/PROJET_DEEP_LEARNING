import gradio as gr
from .pipeline import SentimentPipeline

pipeline = SentimentPipeline()

def run_pipeline(audio_path):
    if audio_path is None:
        return "—", "—", 0.0
    result = pipeline.predict(audio_path)
    if result.error:
        return f"[Erreur] {result.error}", "—", 0.0
    return result.transcription, result.sentiment, result.confidence

demo = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs=[gr.Textbox(label="Transcription"), gr.Label(label="Sentiment"), gr.Number(label="Confiance")],
    title="Détection de sentiment vocal",
)

if __name__ == "__main__":
    demo.launch()