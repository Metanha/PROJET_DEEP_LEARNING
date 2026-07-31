
"""
app_gradio.py
-------------
Interface Gradio stylisée pour la démo du pipeline de détection de sentiment vocal.
"""

import gradio as gr
from app.pipeline import SentimentPipeline

pipeline = SentimentPipeline()

CUSTOM_CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
#title {
    text-align: center;
    margin-bottom: 0;
}
#subtitle {
    text-align: center;
    color: #666;
    margin-top: 0;
    margin-bottom: 20px;
}
#result-box {
    border-radius: 12px;
    padding: 10px;
}
footer {visibility: hidden}
"""

SENTIMENT_EMOJI = {"positif": "🟢 Positif", "négatif": "🔴 Négatif", "neutre": "🟡 Neutre"}


def run_pipeline(audio_path):
    if audio_path is None:
        return "—", "—", 0.0

    result = pipeline.predict(audio_path)

    if result.error:
        return f"⚠️ {result.error}", "—", 0.0

    label = SENTIMENT_EMOJI.get(result.sentiment, result.sentiment)
    return result.transcription, label, round(result.confidence * 100, 1)

demo = gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"), css=CUSTOM_CSS)

with demo:
    gr.Markdown("# 🎙️ Détection de Sentiment Vocal", elem_id="title")
    gr.Markdown(
        "Transcrivez un appel client et détectez automatiquement son sentiment — "
        "propulsé par **Wav2Vec2** et **BERT**.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="🎧 Fichier audio (.wav / .mp3, max. 5 min)",
            )
            with gr.Row():
                clear_btn = gr.ClearButton(value="🗑️ Effacer")
                submit_btn = gr.Button("🚀 Analyser", variant="primary")

        with gr.Column(scale=1, elem_id="result-box"):
            transcription_output = gr.Textbox(
                label="📝 Transcription (ASR)", lines=4, interactive=False
            )
            sentiment_output = gr.Textbox(
                label="💬 Sentiment détecté", interactive=False
            )
            confidence_output = gr.Slider(
                label="📊 Confiance (%)", minimum=0, maximum=100, interactive=False
            )

    clear_btn.add([audio_input, transcription_output, sentiment_output, confidence_output])

    submit_btn.click(
        fn=run_pipeline,
        inputs=audio_input,
        outputs=[transcription_output, sentiment_output, confidence_output],
    )

    gr.Markdown(
        "---\n"
        "ℹ️ Formats acceptés : `.wav`, `.mp3` — durée max. 5 minutes. "
        "Les fichiers invalides (vides, silencieux, format non supporté) "
        "renvoient un message d'erreur explicite plutôt qu'un plantage."
    )

if __name__ == "__main__":
    #demo.launch()
    demo.launch(theme=None ) # ce paramètre n'existe pas, à retirer si tu l'as mis ici

    #demo.launch(theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"))

