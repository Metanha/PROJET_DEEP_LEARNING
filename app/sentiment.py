
"""
benchmark_sentiment.py
-----------------------
Compare 3 modèles de sentiment français/multilingues sur un petit jeu de
phrases annotées représentatives d'un contexte d'appel client.


import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

# --- Jeu de test annoté (à enrichir avec de vraies phrases de ton domaine) ---
TEST_SET = [
    ("je suis très satisfait de votre service, merci beaucoup", "positif"),
    ("c'est vraiment nul, je suis furieux de cette panne", "négatif"),
    ("je voulais juste vérifier le solde de mon compte", "neutre"),
    ("excellent accueil, je recommande vivement", "positif"),
    ("j'attends depuis une heure, c'est inadmissible", "négatif"),
    ("pouvez-vous me confirmer l'adresse de l'agence", "neutre"),
    ("merci infiniment pour votre réactivité", "positif"),
    ("votre produit est défectueux et le service après-vente inutile", "négatif"),
    ("je souhaite modifier mon rendez-vous de la semaine prochaine", "neutre"),
]


@dataclass
class ModelSpec:
    name: str
    hf_repo: str
    label_fn: callable  # transforme la sortie brute du modèle en "positif"/"négatif"/"neutre"


def map_cardiffnlp(probs, id2label):
    idx = int(torch.argmax(probs).item())
    mapping = {"negative": "négatif", "neutral": "neutre", "positive": "positif"}
    return mapping[id2label[idx].lower()]


def map_5stars(probs, id2label):
    idx = int(torch.argmax(probs).item())
    label = id2label[idx]  # ex: "1 star", "5 stars"
    n_stars = int(label.split()[0])
    if n_stars <= 2:
        return "négatif"
    if n_stars == 3:
        return "neutre"
    return "positif"


MODELS = [
    ModelSpec("CardiffNLP XLM-R (tweets, 3 classes natives)", "cardiffnlp/twitter-xlm-roberta-base-sentiment", map_cardiffnlp),
    ModelSpec("nlptown multilingue (5 étoiles)", "nlptown/bert-base-multilingual-uncased-sentiment", map_5stars),
    ModelSpec("DistilCamemBERT (5 étoiles, FR)", "cmarkea/distilcamembert-base-sentiment", map_5stars),
]


def benchmark_model(spec: ModelSpec):
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo)
    model = AutoModelForSequenceClassification.from_pretrained(spec.hf_repo)
    model.eval()
    id2label = model.config.id2label

    y_true, y_pred = [], []
    start = time.time()

    with torch.inference_mode():
        for text, true_label in TEST_SET:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            probs = F.softmax(model(**inputs).logits, dim=-1).squeeze(0)
            pred_label = spec.label_fn(probs, id2label)
            y_true.append(true_label)
            y_pred.append(pred_label)

    elapsed = time.time() - start
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    avg_latency_ms = (elapsed / len(TEST_SET)) * 1000

    return {
        "modèle": spec.name,
        "accuracy": round(acc, 3),
        "f1_macro": round(f1, 3),
        "latence_moy_ms": round(avg_latency_ms, 1),
        "détails": list(zip([t for t, _ in TEST_SET], y_true, y_pred)),
    }


def main():
    results = []
    for spec in MODELS:
        print(f"\n=== {spec.name} ===")
        result = benchmark_model(spec)
        results.append(result)
        for text, true_label, pred_label in result["détails"]:
            marker = "OK" if true_label == pred_label else "FAUX"
            print(f"  [{marker}] attendu={true_label:8s} prédit={pred_label:8s} | {text}")
        print(f"  -> accuracy={result['accuracy']}  f1_macro={result['f1_macro']}  latence={result['latence_moy_ms']}ms/phrase")

    print("\n=== Résumé comparatif ===")
    for r in results:
        print(f"{r['modèle']:45s} acc={r['accuracy']:.3f}  f1={r['f1_macro']:.3f}  latence={r['latence_moy_ms']}ms")


if __name__ == "__main__":
    main()
"""

"""
sentiment.py
------------
Analyse de sentiment via CardiffNLP XLM-RoBERTa (3 classes natives).

Choix justifié par un benchmark comparatif (voir benchmark_sentiment.py et
README) face à deux alternatives (nlptown 5 étoiles, DistilCamemBERT 5
étoiles) : CardiffNLP obtient accuracy=1.0 et f1_macro=1.0 sur un jeu de
test de 9 phrases représentatives d'un contexte d'appel client, contre
~0.78 pour les deux autres, qui confondent plus souvent la classe neutre
avec positif/négatif (ambiguïté de la note "3 étoiles").
Limite connue : entraîné sur des tweets multilingues, pas spécifiquement
sur du français ni sur des transcriptions d'appels — à valider sur un plus
grand jeu de test réel avant mise en production.
"""

#from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_SENTIMENT_MODEL = "Souvikcmsa/BERT_sentiment_analysis"

# mapping des labels natifs du modèle (ex: "negative", "neutral", "positive")
# vers les labels français utilisés dans tout le projet
LABEL_MAPPING = {"negative": "négatif", "neutral": "neutre", "positive": "positif"}


@dataclass
class SentimentResult:
    label: str
    confidence: float


class SentimentModel:
    def __init__(self, model_name: str = DEFAULT_SENTIMENT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def predict(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(label="neutre", confidence=0.0)

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)

        best_idx = int(torch.argmax(probs).item())
        confidence = float(probs[best_idx].item())
        raw_label = self.id2label[best_idx].lower()
        label = LABEL_MAPPING.get(raw_label, "neutre")

        return SentimentResult(label=label, confidence=round(confidence, 4))

"""
benchmark_sentiment_en.py
---------------------------
Compare 4 modèles BERT anglais pour la classification de sentiment.
"""
"""
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

TEST_SET = [
    ("i am very satisfied with your service, thank you so much", "positif"),
    ("this is really terrible, i'm furious about this outage", "négatif"),
    ("i just wanted to check my account balance", "neutre"),
    ("excellent service, i highly recommend it", "positif"),
    ("i've been waiting for an hour, this is unacceptable", "négatif"),
    ("can you confirm the branch address please", "neutre"),
    ("thank you so much for your quick response", "positif"),
    ("your product is defective and customer support is useless", "négatif"),
    ("i'd like to reschedule my appointment for next week", "neutre"),
]


@dataclass
class ModelSpec:
    name: str
    hf_repo: str
    label_fn: callable


def map_5stars(probs, id2label):
    idx = int(torch.argmax(probs).item())
    n_stars = int(id2label[idx].split()[0])
    if n_stars <= 2:
        return "négatif"
    if n_stars == 3:
        return "neutre"
    return "positif"


def map_generic_3class(probs, id2label):
    idx = int(torch.argmax(probs).item())
    raw = id2label[idx].lower()
    if "neg" in raw or raw in ("0", "label_0"):
        return "négatif"
    if "neu" in raw or raw in ("1", "label_1"):
        return "neutre"
    if "pos" in raw or raw in ("2", "label_2"):
        return "positif"
    return f"INCONNU({raw})"


MODELS = [
    ModelSpec("nlptown mBERT (5 stars)", "nlptown/bert-base-multilingual-uncased-sentiment", map_5stars),
    ModelSpec("mervp SentimentBERT", "mervp/SentimentBERT", map_generic_3class),
    ModelSpec("Souvikcmsa BERT", "Souvikcmsa/BERT_sentiment_analysis", map_generic_3class),
    ModelSpec("MarieAngeA13 BERT", "MarieAngeA13/Sentiment-Analysis-BERT", map_generic_3class),
]


def benchmark_model(spec: ModelSpec):
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo)
    model = AutoModelForSequenceClassification.from_pretrained(spec.hf_repo)
    model.eval()
    id2label = model.config.id2label
    print(f"  id2label brut : {id2label}")

    y_true, y_pred = [], []
    start = time.time()
    with torch.inference_mode():
        for text, true_label in TEST_SET:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            probs = F.softmax(model(**inputs).logits, dim=-1).squeeze(0)
            pred_label = spec.label_fn(probs, id2label)
            y_true.append(true_label)
            y_pred.append(pred_label)
    elapsed = time.time() - start

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    latency = (elapsed / len(TEST_SET)) * 1000
    return acc, f1, latency, list(zip([t for t, _ in TEST_SET], y_true, y_pred))


def main():
    results = []
    for spec in MODELS:
        print(f"\n=== {spec.name} ===")
        acc, f1, latency, details = benchmark_model(spec)
        results.append((spec.name, acc, f1, latency))
        for text, true_label, pred_label in details:
            marker = "OK" if true_label == pred_label else "FAUX"
            print(f"  [{marker}] attendu={true_label:8s} prédit={pred_label:8s} | {text}")
        print(f"  -> accuracy={acc:.3f}  f1_macro={f1:.3f}  latence={latency:.1f}ms/phrase")

    print("\n=== Résumé comparatif ===")
    for name, acc, f1, latency in results:
        print(f"{name:30s} acc={acc:.3f}  f1={f1:.3f}  latence={latency:.1f}ms")


if __name__ == "__main__":
    main()
    
"""