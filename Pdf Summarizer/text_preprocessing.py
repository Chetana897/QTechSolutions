#text_preprocessing.py
import re


def preprocess_text(text):

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]+\]', '', text)
    return text.strip()
