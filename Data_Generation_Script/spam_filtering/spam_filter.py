import os
import pandas as pd
import random
import string

# List of cuss words and their variations
cuss_words = [
    "fuck", "shit", "bitch", "asshole", "motherfucker", "bastard", "dick", "cunt", "cock", "pussy",
    "ass", "damn", "hell", "fucker", "shithead", "douchebag", "dickhead", "asshat", "twat", "wanker",
    "arse", "bollocks", "bugger", "crap", "bloody", "git", "knob", "prick", "sod", "tosser",
    "wank", "arsehole", "bellend", "fanny", "minge", "nutter", "slag", "spastic", "tart", "tit"
]

# List of common spam keywords and their variations
spam_keywords = [
    "sale", "free", "discount", "buy", "now", "limited", "offer", "promotion", "money", "cash",
    "click", "here", "guaranteed", "urgent", "act", "fast", "order", "now", "bonus", "prize",
    "win", "gift", "card", "click", "below", "unsubscribe", "shop", "visit", "website", "online",
    "earn", "income", "investment", "opportunity", "work", "from", "home", "earnings", "profit",
    "income", "investment", "rich", "wealth", "cash", "easy", "money", "secret", "method"
]

# Function to generate variations of words with intentional typos
def generate_variations(word):
    variations = []
    for i in range(1, random.randint(1, 4)):
        variation = word
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, len(variation) - 1)
            variation = variation[:idx] + random.choice(string.ascii_lowercase) + variation[idx + 1:]
        variations.append(variation)
    return variations

# Generate variations of cuss words and spam keywords
word_variations = []
for word in cuss_words + spam_keywords:
    word_variations.extend(generate_variations(word))

# Shuffle the variations
random.shuffle(word_variations)

# Function to generate a dataset for spam filtering
def generate_spam_filtering_dataset(size):
    data = {'text': []}
    for _ in range(size):
        # Choose a random word or variation
        word = random.choice(word_variations)
        # Introduce random typos
        for _ in range(random.randint(0, 2)):
            idx = random.randint(0, len(word) - 1)
            word = word[:idx] + random.choice(string.ascii_lowercase) + word[idx + 1:]
        data['text'].append(word)
    return pd.DataFrame(data)

# Define base directory where datasets will be stored
base_directory = '../Spam_Filtering_Datasets'

# Create the directory if it doesn't exist
os.makedirs(base_directory, exist_ok=True)

# Generate and save multiple datasets
for i in range(1, 21):
    # Define directory for the current dataset
    dataset_directory = os.path.join(base_directory, f'Dataset{i}')
    os.makedirs(dataset_directory, exist_ok=True)
    
    # Generate the dataset
    dataset_size = 10000  # Adjust the size as needed
    spam_dataset = generate_spam_filtering_dataset(dataset_size)
    
    # Save the dataset to a CSV file within the dataset directory
    csv_filename = os.path.join(dataset_directory, f'spam_filtering_dataset_{i}.csv')
    spam_dataset.to_csv(csv_filename, index=False)
    
    print(f'Dataset {i} saved to: {csv_filename}')
