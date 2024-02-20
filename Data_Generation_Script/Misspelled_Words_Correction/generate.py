import pandas as pd
import random
import string

# Function to introduce random typos in a word
def introduce_typos(word, typo_rate=0.2):
    """
    Introduce typos in a word.
    Args:
        word (str): The word to introduce typos into.
        typo_rate (float): The rate of typos to introduce.
    Returns:
        str: The word with introduced typos.
    """
    if random.random() < typo_rate:
        # Replace a random character with another random character
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + random.choice(string.ascii_lowercase) + word[idx + 1:]
    else:
        return word

# Function to generate a larger dataset with misspelled words and more correct spellings
def generate_large_dataset(correct_words, size, typo_rate=0.2):
    """
    Generate a larger dataset with misspelled words and more correct spellings.
    Args:
        correct_words (list): List of correctly spelled words.
        size (int): Size of the dataset to generate.
        typo_rate (float): The rate of typos to introduce.
    Returns:
        pandas.DataFrame: DataFrame containing 'misspelled_word' and 'correct_word'.
    """
    data = {'misspelled_word': [], 'correct_word': []}
    for _ in range(size):
        correct_word = random.choice(correct_words)
        misspelled_word = introduce_typos(correct_word, typo_rate)
        data['misspelled_word'].append(misspelled_word)
        data['correct_word'].append(correct_word)
    return pd.DataFrame(data)

# List of correctly spelled words (expanded list)
correct_words = ["learning", "machine", "intelligence", "python", "programming", "language", "algorithm", "data", "science",
                 "model", "analysis", "deep", "neural", "network", "framework", "development", "application", "computer",
                 "vision", "natural", "processing", "artificial", "intelligence", "supervised", "unsupervised", "classification",
                 "regression", "clustering", "dimensionality", "reduction", "feature", "extraction", "hyperparameter",
                 "optimization", "ensemble", "learning", "convolutional", "recurrent", "sequence", "generation", "gpt", "chatbot"]

# Generate a larger dataset with 500,000 samples and 0.2 typo rate
large_dataset = generate_large_dataset(correct_words, size=500000, typo_rate=0.2)

# Save the larger dataset to a CSV file
large_dataset.to_csv('spell_correction_large_data.csv', index=False)
