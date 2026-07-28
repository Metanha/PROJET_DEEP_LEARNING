
"""
audio = AudioSegment.from_file("tests/samples/SPEAKER_00.wav")
print("Canaux:", audio.channels)
print("Fréquence:", audio.frame_rate)
print("Durée (s):", len(audio) / 1000)




Prétraitement des fichiers audio avant transcription.

Étapes réalisées :
Chargement du fichier (.wav ou .mp3) avec `pydub` (gère les deux formats).
Conversion en mono (moyenne des canaux si stéréo).
Rééchantillonnage à 16 kHz (fréquence attendue par Wav2Vec 2.0).
Normalisation de l'amplitude (peak normalization dans [-1, 1]).
Vérifications de validité (fichier vide, durée nulle, silence total).

"""

#from __future__ import annotations

import os
import numpy as np
import torch
import torchaudio
import imageio_ffmpeg
from pydub import AudioSegment
from pydub.utils import which


AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

AudioSegment.ffprobe = imageio_ffmpeg.get_ffmpeg_exe()

TARGET_SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 5 * 60  # 5 minutes durée maximale pour un fichier audio
SUPPORTED_EXTENSIONS = {".wav", ".mp3"} # extensions de fichiers audio supportées pour le prétraitement

import os
# module standard pour vérifier l'existence d'un fichier, sa taille, son extension

SUPPORTED_EXTENSIONS = {".wav", ".mp3"}
# ensemble des extensions autorisées par le cahier des charges — un set (pas une liste)
# car le test d'appartenance "in" est en O(1) au lieu de O(n)


class AudioValidationError(Exception):
    """Erreur levée quand un fichier audio est invalide."""
    # on crée une exception personnalisée plutôt que de lever une ValueError générique :
    # ça permet à l'appelant (pipeline, API) de savoir précisément qu'il s'agit
    # d'un problème de validation audio, et pas d'une autre erreur (bug, réseau, etc.)


def validate_audio_file(path: str) -> None:
    # ne retourne rien (None) : 
    # soit elle lève une exception. 
    # une fonction qui traite les fichiers invalides en levant des exceptions est plus simple à utiliser que de renvoyer un booléen

    if not os.path.exists(path):
        # os.path.exists renvoie False si le chemin n'existe pas du tout
        # traite le cas d'un fichier introuvable (mauvais chemin, fichier supprimé, etc.)
        raise AudioValidationError(f"Fichier introuvable : {path}")

    ext = os.path.splitext(path)[1].lower()
    # os.path.splitext("a/b.WAV") -> ("a/b", ".WAV") ; [1] prend l'extension
    # .lower() pour que ".WAV" et ".wav" soient traités pareil

    if ext not in SUPPORTED_EXTENSIONS:
        #les fichiers de mauvaises extensions sont traités ici, on lève une exception si l'extension n'est pas dans la liste des formats supportés
        raise AudioValidationError(
            f"Format non supporté '{ext}'. Formats acceptés : {SUPPORTED_EXTENSIONS}"
        )
        
    if os.path.getsize(path) == 0:    
        # la condition traite le cas d'un fichier vide (0 octets)
        raise AudioValidationError("Le fichier audio est vide.")


def _load_with_pydub(path: str):
    # cette fonction charge le fichier audio avec pydub, le convertit en mono si nécessaire

    audio = AudioSegment.from_file(path)
    
    sample_rate = audio.frame_rate
    
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    # get_array_of_samples() renvoie les échantillons bruts en entiers (int16 en général)
    # np.array() les charge dans un tableau numpy
    # .astype(np.float32) convertit en flottant AVANT toute division,
    
    if audio.channels > 1:
        # si le fichier est stéréo (ou plus) au lieu de mono
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
        
    max_val = float(1 << (8 * audio.sample_width - 1))
    samples = samples / max_val
    # normalisation depuis l'échelle entière brute vers une échelle flottante [-1, 1]
    return samples, sample_rate
    # on renvoie un tuple : le signal ET sa fréquence d'origine 


def preprocess_audio(path: str) -> torch.Tensor:
    validate_audio_file(path)
    # première chose faite : on rejette tout de suite les fichiers invalides,
    # avant de perdre du temps à les charger

    samples, sample_rate = _load_with_pydub(path)
    # déballe le tuple renvoyé par la fonction privée

    if samples.size == 0:
        # .size est le nombre total d'éléments du tableau numpy
        raise AudioValidationError("Signal audio vide après lecture.")
        # cas différent de "fichier vide" (taille 0 octet) : ici le fichier existe
        # et a une taille non nulle, mais pydub n'a extrait aucun échantillon

    duration = len(samples) / sample_rate
    # nombre d'échantillons divisé par la fréquence

    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(f"Durée trop longue ({duration:.1f}s).")
    
    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)

    if sample_rate != TARGET_SAMPLE_RATE:
        # on ne rééchantillonne que si nécessaire, pour ne pas dégrader
        # inutilement un signal déjà à la bonne fréquence

        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
        # crée un objet "transformation" configuré pour convertir
        # de la fréquence d'origine vers 16 kHz
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)
    # opération inverse de unsqueeze : (1, N) -> (N,)
    # on repasse en 1D car c'est ce que le modèle ASR attend en entrée
    peak = waveform.abs().max()
    # valeur absolue maximale du signal = l'amplitude la plus forte (positive ou négative)
    if peak == 0:
        raise AudioValidationError("Audio silencieux : amplitude nulle.")
        # si le pic est 0, tout le signal est plat 
    waveform = waveform / peak
    # peak normalization : on divise tout le signal par son maximum,
    # ce qui ramène l'amplitude max exactement à 1.0 

    return waveform
    # tenseur final : 1D, float32, 16 kHz, amplitude normalisée dans [-1, 1]