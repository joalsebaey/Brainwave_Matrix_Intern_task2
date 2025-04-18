"""
Text preprocessing functions for social media sentiment analysis
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    """Download required NLTK resources"""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)

# Call this function to ensure resources are available
download_nltk_resources()

def clean_text(text):
    """
    Clean and normalize text data.
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove user mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """
    Tokenize text into words.
    
    Args:
        text (str): Cleaned text
        
    Returns:
        list: List of tokens
    """
    return word_tokenize(text)

def remove_stopwords(tokens):
    """
    Remove common stopwords from tokens.
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: Filtered tokens
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    """
    Lemmatize tokens to their root form.
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: Lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    """
    Complete text preprocessing pipeline.
    
    Args:
        text (str): Raw text
        
    Returns:
        list: Processed tokens
        str: Processed text
    """
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)
    processed_text = ' '.join(lemmatized_tokens)
    
    return lemmatized_tokens, processed_text
