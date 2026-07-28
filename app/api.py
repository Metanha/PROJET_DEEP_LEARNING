import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .pipeline import SentimentPipeline
from .preprocessing import SUPPORTED_EXTENSIONS

app = FastAPI(title="API Détection Sentiment Vocal")
pipeline = SentimentPipeline()  # chargé une seule fois au démarrage du serveur


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Format non supporté '{ext}'. Attendu : {sorted(SUPPORTED_EXTENSIONS)}"},
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = pipeline.predict(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # nettoyage systématique, même en cas d'erreur

    status_code = 422 if result.error else 200
    return JSONResponse(status_code=status_code, content=result.to_dict())