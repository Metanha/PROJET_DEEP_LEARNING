
## 1. Modèles utilisés et justification

| Tâche | Modèle | Lien | Justification |
|---|---|---|---|
| ASR | Wav2Vec 2.0 XLSR-53 anglais | [jonatasgrosman/wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) | Fine-tuné sur Common Voice EN, robuste grâce au pré-entraînement multilingue XLSR-53, gratuit et prêt à l'emploi. |
| Sentiment | BERT (3 classes natives) | [Souvikcmsa/BERT_sentiment_analysis](https://huggingface.co/Souvikcmsa/BERT_sentiment_analysis) | Retenu après un benchmark comparatif de 4 modèles BERT (voir ci-dessous). Loss documentée de 0.499, accuracy 0.799 sur sa fiche modèle, et meilleur score sur notre propre jeu de test. |

### Benchmark comparatif du modèle de sentiment

| Modèle | Accuracy | F1 macro | Latence | Notes |
|---|---|---|---|---|
| nlptown/bert-base-multilingual-uncased-sentiment | 0.889 | 0.886 | 200ms | mBERT, sortie 5 étoiles mappée en 3 classes |
| mervp/SentimentBERT | 0.889 | 0.886 | 50ms | BERT, 3 classes natives |
| **Souvikcmsa/BERT_sentiment_analysis** | **1.000** | **1.000** | **49ms** | BERT, 3 classes natives, loss=0.499 documentée |
| MarieAngeA13/Sentiment-Analysis-BERT | 1.000 | 1.000 | 47ms | BERT, 3 classes natives, pas de métriques publiées |

Souvikcmsa retenu à égalité de score avec MarieAngeA13, en s'appuyant sur la loss/accuracy
documentée sur sa fiche Hugging Face comme critère de départage supplémentaire.


## 2. Installation

```bash
git clone <url-du-depot>
cd <nom-du-depot>

python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

`ffmpeg` est requis pour lire les fichiers `.mp3` :
- Linux (conteneur/CI) : `apt-get install ffmpeg`
- Windows (environnement conda) : `conda install -c conda-forge ffmpeg`
- macOS : `brew install ffmpeg`

Au premier lancement, les modèles Hugging Face sont téléchargés automatiquement
(~1.2 Go pour Wav2Vec2 XLSR-53 anglais, ~440 Mo pour le modèle de sentiment BERT).

## 3. Utilisation

### Interface Gradio

```bash
python app_gradio.py
```

Ouvre `http://127.0.0.1:7860`. Upload ou enregistrement d'un audio, affichage de la
transcription intermédiaire puis du sentiment détecté avec score de confiance.

### API REST

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

La racine `/` redirige vers la documentation interactive `/docs`, qui permet de tester
`/predict` directement dans le navigateur.

**Exemple curl :**
```bash
curl.exe -X POST "http://localhost:8000/predict" -F "file=@tests/samples/positive.wav"
```

**Exemple Python :**
```python
import requests

with open("tests/samples/positive.wav", "rb") as f:
    r = requests.post("http://localhost:8000/predict", files={"file": f})
print(r.json())
# {"transcription": "...", "sentiment": "positif", "confidence": 0.94, "error": null}
```

### Docker

```bash
docker build -t sentiment-vocal .
docker run -p 8000:8000 sentiment-vocal
```

### Démo en ligne 

[Lien Hugging Face Spaces à compléter une fois déployé]

## 4. Cas d'usage

- Centre d'appel : détection automatique de l'insatisfaction client pour priorisation
  des escalades.
- Analyse de tendances : agrégation du sentiment sur un grand volume d'appels.
- Intégration dans un CRM via l'API `/predict`.

## 5. Limites connues

- **Langue** : pipeline anglais uniquement (voir section 3) ;
- **Longueur des appels et troncature** : le tokenizer du modèle de sentiment tronque à
  512 tokens. Sur un appel long (3-5 minutes, plusieurs centaines de mots), seule
  la première partie de la transcription est effectivement analysée, ce qui peut biaiser
  le résultat si le sentiment évolue en cours d'appel.
- **Pas de diarisation** : le sentiment est calculé sur l'ensemble de la transcription,
  sans distinguer client/agent.
- **Qualité de l'ASR** : peut se dégrader fortement sur de l'audio bruité, avec un fort
  accent, ou mal décodé.
- **Durée max. 5 minutes par fichier** (contrainte du cahier des charges) — au-delà, le
  fichier est rejeté proprement avec un message d'erreur.

## 6. Gestion des erreurs

Le pipeline intercepte et retourne proprement, sans crash :
- format de fichier non supporté (autre que `.wav`/`.mp3`) ;
- fichier vide ou introuvable ;
- audio silencieux (amplitude nulle) ;
- durée excédant 5 minutes ;
- transcription vide après ASR.

## 7. Tests

```bash
pytest tests/ -v
```

## 8. Évaluation quantitative

```bash
python evaluate_sentiment.py
```

Compare l'accuracy/F1 du modèle de sentiment seul (sur texte de référence propre) à
celle du pipeline complet (audio réel → ASR → sentiment), pour isoler l'impact des
erreurs de transcription sur le résultat final.

