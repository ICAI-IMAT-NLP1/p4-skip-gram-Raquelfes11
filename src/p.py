from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize

def load_and_preprocess_data(infile: str) -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        infile (str): The path to the input file containing text data.

    Returns:
        List[str]: A list of preprocessed and tokenized words from the input text.
    """
    with open(infile,"r",encoding="utf-8") as file:
        text = file.read().lower()  # Read the entire file

    # Preprocess and tokenize the text
    # TODO
    tokens: List[str] = tokenize(text)

    return tokens


tokens = load_and_preprocess_data("data/text8")
print(tokens)