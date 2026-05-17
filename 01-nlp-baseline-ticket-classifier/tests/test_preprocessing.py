import sys
import os
sys.path.append(os.path.abspath('../'))
import pytest
from src.preprocessing import clean_text, stem_text

def test_clean_text_lowercasing():
    assert clean_text("CRÍTICO Servidor") == "crítico servidor"

def test_clean_text_extra_spaces():
    assert clean_text("Anfrage    zur   Verfügbarkeit") == "anfrage zur verfügbarkeit"

def test_clean_text_html_tags():
    assert clean_text("Sehr geehrter <name>,\n\nvielen Dank") == "sehr geehrter vielen dank"

def test_clean_text_multilingual_characters():
    # Test if preserves multilingual characters for de, es, pt, fr
    text_input = "äöüß éèàç ñí"
    assert clean_text(text_input) == "äöüß éèàç ñí"

def test_clean_text_empty_and_invalid():
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_clean_text_all_pii_tags():
    assert clean_text("Contact: <email> or <phone>") == "contact or"

def test_stem_text():
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    assert stem_text("running", stemmer) == "run"
    assert stem_text("happily", stemmer) == "happili"
    assert stem_text("studies", stemmer) == "studi"