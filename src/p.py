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

def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    # TODO
    word_counts: Counter = Counter(words)
    # Sorting the words from most to least frequent in text occurrence.
    sorted_words:List[str] = [word for word, _ in word_counts.most_common()]
    sorted_vocab: List[int] = [words.index(word) for word in sorted_words]

    # Create int_to_vocab and vocab_to_int dictionaries.
    int_to_vocab: Dict[int, str] = dict()
    i:int = 0
    for index in sorted_vocab:
        int_to_vocab[i] = words[index]
        i += 1

    vocab_to_int: Dict[str, int] = {word: i for i, word in int_to_vocab.items()}
    print(vocab_to_int)
    print(int_to_vocab)
    print(sorted_vocab)
    return vocab_to_int, int_to_vocab







#tokens = load_and_preprocess_data("data/text8")
create_lookup_tables(["hello", "world", "hello", "test"])