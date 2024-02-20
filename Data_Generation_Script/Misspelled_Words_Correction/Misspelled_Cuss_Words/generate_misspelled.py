import os
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

# Larger list of correctly spelled words
correct_words = [
    "learning", "machine", "intelligence", "python", "programming", "language", "algorithm", "data", "science",
    "model", "analysis", "deep", "neural", "network", "framework", "development", "application", "computer",
    "vision", "natural", "processing", "artificial", "intelligence", "supervised", "unsupervised", "classification",
    "regression", "clustering", "dimensionality", "reduction", "feature", "extraction", "hyperparameter",
    "optimization", "ensemble", "learning", "convolutional", "recurrent", "sequence", "generation", "gpt", "chatbot",
    "statistics", "probability", "distribution", "sampling", "estimation", "inference", "hypothesis", "testing",
    "confidence", "interval", "p-value", "correlation", "covariance", "regression", "anova", "chi-square", "test",
    "normal", "binomial", "poisson", "uniform", "exponential", "student's", "t-distribution", "central", "limit", 
    "theorem", "maximum", "likelihood", "bayesian", "mcmc", "monte", "carlo", "simulation", "random", "forest", 
    "decision", "tree", "k-nearest", "neighbors", "support", "vector", "machine", "neural", "network", "activation", 
    "function", "backpropagation", "gradient", "descent", "dropout", "batch", "normalization", "relu", "sigmoid", 
    "softmax", "training", "testing", "validation", "overfitting", "underfitting", "cross-validation", "ensemble",
    "bagging", "boosting", "adaboost", "gradient", "boosting", "xgboost", "lightgbm", "catboost", "autoencoder", 
    "variational", "autoencoder", "gan", "generative", "adversarial", "network", "image", "recognition", "object", 
    "detection", "segmentation", "semantic", "instance", "style", "transfer", "reinforcement", "q-learning", "policy", 
    "iteration", "value", "function", "deep", "deterministic", "policy", "gradients", "actor-critic", "temporal", 
    "difference", "dqn", "double", "q-learning", "experience", "replay", "policy", "gradients", "asynchronous", 
    "advantage", "actor-critic", "trust", "region", "policy", "optimization", "ppo", "proximal", "policy", "optimization", 
    "dppo", "distributed", "deep", "deterministic", "policy", "gradients", "d4pg", "distributional", "dqn", "dueling", 
    "network", "soft", "actor-critic", "td3", "twin", "delayed", "dqn", "sac", "soft", "actor-critic", "clustering", 
    "k-means", "hierarchical", "dbscan", "mean", "shift", "density", "estimation", "em", "gaussian", "mixture", 
    "model", "expectation-maximization", "independent", "component", "analysis", "dimensionality", "reduction", 
    "pca", "principal", "component", "analysis", "svd", "singular", "value", "decomposition", "ica", "independent", 
    "component", "analysis", "nmf", "non-negative", "matrix", "factorization", "t-sne", "t-distributed", "stochastic", 
    "neighbor", "embedding", "association", "rule", "apriori", "fp-growth", "eclat", "collaborative", "filtering", 
    "user-based", "item-based", "content-based", "matrix", "factorization", "deep", "belief", "network", "restricted", 
    "boltzmann", "machine", "markov", "model", "hidden", "markov", "model", "hmm", "conditional", "random", "field", 
    "crf", "graphical", "model", "graph", "neural", "network", "graph", "convolutional", "network", "graph", "attention", 
    "network", "graph", "autoencoder", "graph", "embedding", "network", "graph", "reinforcement", "learning", 
    "knowledge", "representation", "embedding", "network", "one-hot", "encoding", "word", "embedding", "word2vec", 
    "skip-gram", "cbow", "glove", "fasttext", "bert", "attention", "transformer", "sequence", "to", "sequence", 
    "seq2seq", "encoder", "decoder", "transformer", "architecture", "attention", "mechanism", "self-attention", 
    "multi-head", "attention", "bidirectional", "attention", "language", "model", "pre-training", "fine-tuning", 
    "masked", "language", "modeling", "mlm", "next", "sentence", "prediction", "nsp", "token", "classification", 
    "question", "answering", "squad", "natural", "language", "inference", "nli", "sentiment", "analysis", 
    "named", "entity", "recognition", "ner", "part-of-speech", "tagging", "pos", "dependency", "parsing", "chunking", 
    "syntax", "semantic", "role", "labeling", "parsing", "treebank", "wordnet", "stopwords", "stemming", "lemmatization", 
    "tokenization", "bag-of-words", "tf-idf", "word", "embedding", "sequence", "embedding", "transformer", "encoder", 
    "recurrent", "neural", "network", "lstm", "long", "short", "term", "memory", "gru", "gated", "recurrent", "unit", 
    "bidirectional", "lstm", "attention", "lstm", "convolutional", "neural", "network", "cnn", "recurrent", "neural", 
    "network", "rnn", "attention", "mechanism", "transformer", "bidirectional", "attention", "transformer", "seq2seq", 
    "neural", "machine", "translation", "image", "recognition", "object", "detection", "segmentation", "classification", 
    "localization", "instance", "segmentation", "semantic", "segmentation", "depth", "prediction", "pose", "estimation", 
    "image", "generation", "super", "resolution", "style", "transfer", "gan", "generative", "adversarial", "network", 
    "conditional", "gan", "cgan", "cycle", "gan", "pix2pix", "discriminator", "generator", "encoder", "decoder", "autoencoder", 
    "variational", "autoencoder", "vae", "conditional", "variational", "autoencoder", "cva", "anomaly", "detection", 
    "ensemble", "method", "isolation", "forest", "local", "outlier", "factor", "lof", "minimum", "covariance", "detection", 
    "one-class", "svm", "oc-svm", "support", "vector", "data", "description", "k-nearest", "neighbors", "clustering", 
    "density", "based", "spatial", "clustering", "spectral", "clustering", "topic", "modeling", "lda", "latent", "dirichlet", 
    "allocation", "nmf", "non-negative", "matrix", "factorization", "lsa", "latent", "semantic", "analysis", "hdp", "hierarchical", 
    "dirichlet", "process", "network", "analysis", "network", "centrality", "degree", "centrality", "closeness", "centrality", 
    "betweenness", "centrality", "eigenvector", "centrality", "pagerank", "community", "detection", "modularity", "louvain", 
    "community", "detection", "label", "propagation", "community", "detection", "conductance", "clustering", "cohesion", 
    "clustering", "connectivity", "clustering", "entropy", "clustering", "evaluation", "silhouette", "coefficient", "calinski", 
    "harabasz", "index", "davies-bouldin", "index", "fowlkes-mallows", "index", "rand", "index", "adjusted", "rand", 
    "index", "mutual", "information", "adjusted", "mutual", "information", "homogeneity", "completeness", "v-measure", 
    "confusion", "matrix", "accuracy", "precision", "recall", "f1-score", "roc", "curve", "auc", "area", "under", 
    "curve", "precision-recall", "curve", "average", "precision", "mean", "average", "precision", "at", "k", "kappa", 
    "cohen's", "kappa", "cohens-kappa", "intra-annotator", "agreement", "inter-annotator", "agreement", "bias-variance", 
    "tradeoff", "overfitting", "underfitting", "cross-validation", "holdout", "k-fold", "stratified", "shuffle", 
    "leave-one-out", "bootstrap", "resampling", "hyperparameter", "tuning", "grid", "search", "random", "search", 
    "bayesian", "optimization", "genetic", "algorithm", "learning", "curve", "validation", "curve", "bias", 
    "variance", "decomposition", "confusion", "matrix", "roc", "auc", "pr", "auc", "f1-score", "precision", 
    "recall", "accuracy", "overfitting", "underfitting", "cross-validation", "holdout", "k-fold", "bootstrap", 
    "resampling", "bias-variance", "tradeoff", "hyperparameter", "tuning", "grid", "search", "random", "search", 
    "bayesian", "optimization", "ensemble", "learning", "boosting", "bagging", "stacking", "meta-learning", 
    "random", "forest", "gradient", "boosting", "xgboost", "lightgbm", "catboost", "unsupervised", "learning", 
    "clustering", "k-means", "hierarchical", "dbscan", "mean", "shift", "density", "estimation", "em", 
    "gaussian", "mixture", "model", "expectation-maximization", "non-negative", "matrix", "factorization", 
    "nmf", "singular", "value", "decomposition", "svd", "latent", "dirichlet", "allocation", "lda", "graph", 
    "embedding", "deepwalk", "node2vec", "word2vec", "bert", "pre-training", "fine-tuning", "elmo", "gpt", 
    "transformer", "attention", "mechanism", "self-attention", "multi-head", "attention", "bert", "architecture", 
    "language", "modeling", "masked", "language", "modeling", "mlm", "next", "sentence", "prediction", "nsp", 
    "question", "answering", "squad", "named", "entity", "recognition", "ner", "part-of-speech", "tagging", 
    "pos", "dependency", "parsing", "syntax", "semantic", "role", "labeling", "parsing", "treebank", "wordnet", 
    "stopwords", "stemming", "lemmatization", "tokenization", "bag-of-words", "tf-idf", "word", "embedding", 
    "sequence", "embedding", "transformer", "encoder", "recurrent", "neural", "network", "lstm", "gru", 
    "bidirectional", "lstm", "convolutional", "neural", "network", "cnn", "recurrent", "neural", "network", 
    "rnn", "attention", "mechanism", "transformer", "seq2seq", "neural", "machine", "translation", "image", 
    "recognition", "object", "detection", "segmentation", "classification", "localization", "instance", 
    "segmentation", "semantic", "segmentation", "depth", "prediction", "pose", "estimation", "image", 
    "generation", "super", "resolution", "style", "transfer", "gan", "generative", "adversarial", "network", 
    "conditional", "gan", "cycle", "gan", "pix2pix", "autoencoder", "variational", "autoencoder", "vae", 
    "conditional", "variational", "autoencoder", "cva", "anomaly", "detection", "isolation", "forest", 
    "local", "outlier", "factor", "lof", "minimum", "covariance", "detection", "one-class", "svm", "oc-svm", 
    "support", "vector", "data", "description", "k-nearest", "neighbors", "clustering", "density", "based", 
    "spatial", "clustering", "spectral", "clustering", "topic", "modeling", "latent", "dirichlet", "allocation", 
    "nmf", "non-negative", "matrix", "factorization", "lsa", "latent", "semantic", "analysis", "hdp", 
    "hierarchical", "dirichlet", "process", "network", "analysis", "network", "centrality", "degree", "centrality", 
    "closeness", "centrality", "betweenness", "centrality", "eigenvector", "centrality", "pagerank", "community", 
    "detection", "modularity", "louvain", "community", "detection", "label", "propagation", "community", "detection", 
    "conductance", "clustering", "cohesion", "clustering", "connectivity", "clustering", "entropy", "clustering", 
    "evaluation", "silhouette", "coefficient", "calinski", "harabasz", "index", "davies-bouldin", "index", "fowlkes-mallows", 
    "index", "rand", "index", "adjusted", "rand", "index", "mutual", "information", "adjusted", "mutual", "information", 
    "homogeneity", "completeness", "v-measure", "confusion", "matrix", "accuracy", "precision", "recall", "f1-score", 
    "roc", "curve", "auc", "area", "under", "curve", "precision-recall", "curve", "average", "precision", "mean", 
    "average", "precision", "at", "k", "kappa", "cohen's", "kappa", "cohens-kappa", "intra-annotator", "agreement", 
    "inter-annotator", "agreement", "bias-variance", "tradeoff", "overfitting", "underfitting", "cross-validation", 
    "holdout", "k-fold", "stratified", "shuffle", "leave-one-out", "bootstrap", "resampling", "hyperparameter", "tuning", 
    "grid", "search", "random", "search", "bayesian", "optimization", "genetic", "algorithm", "learning"]

# Define base directory where datasets will be stored
base_directory = 'LoneWolfData/Misspelled_Words_Correction'

# Generate datasets and save to corresponding directories
for i in range(1, 21):
    # Generate large dataset
    large_dataset = generate_large_dataset(correct_words, size=500000, typo_rate=0.2)
    
    # Define directory for the current dataset
    dataset_directory = os.path.join(base_directory, f'Dataset{i}')
    
    # Create the directory if it doesn't exist
    os.makedirs(dataset_directory, exist_ok=True)
    
    # Save the dataset to CSV file within the dataset directory
    csv_filename = os.path.join(dataset_directory, f'spell_correction_large_data_{i}.csv')
    large_dataset.to_csv(csv_filename, index=False)

    print(f'Dataset {i} saved to: {csv_filename}')
