from mlOCR.utils.common import load_text_file
import os
from spellchecker import SpellChecker
from pathlib import Path


class TextPostProcessing:
    def __init__(self, TextPostProcessingConfig):
        self.text_input_path=TextPostProcessingConfig.text_input_path
        self.spell = SpellChecker()

    def correct_spelling(self, text):
        words = text.split()
        corrected = [self.spell.correction(word) if self.spell.correction(word) else word for word in words if word is not None]
        corrected_text=" ".join(corrected)
        return corrected_text

    def get_text(self)->str:
        text=load_text_file(Path(os.path.join(self.text_input_path,'crnn_raw_text.txt')))
        
        return text