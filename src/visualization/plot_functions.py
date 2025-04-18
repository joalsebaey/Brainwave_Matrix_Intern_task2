"""
Feature extraction functions for social media content
"""

from collections import Counter, defaultdict
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def extract_features(df):
    """
    Extract text features from social media posts.
    
    Args:
        df (pandas.DataFrame): Dataset with text
        
    Returns:
        dict: Dictionary of features
    """
    features = {}
    
    # Get processed text
    all_processed_text = ' '.join(df['processed_text'].dropna())
    
    # Word frequency distribution
    all_tokens = []
    for tokens in df['tokens'].dropna():
        if isinstance(tokens, list):
            all_tokens.extend(tokens)
    
    word_freq = Counter(all_tokens)
    features['top_words'] = dict(word_freq.most_common(50))
    
    # Extract n-grams
    bigrams = list(ngrams(all_tokens, 2))
    trigrams = list(ngrams(all_tokens, 3))
    
    features['top_bigrams'] = dict(Counter(bigrams).most_common(30))
    features['top_trigrams'] = dict(Counter(trigrams).most_common(20))
    
    # Extract hashtags
    hashtags = []
    if 'Hashtags' in df.columns:
        for hashtag_str in df['Hashtags'].dropna():
            if isinstance(hashtag_str, str):
                tags = [tag.strip() for tag in hashtag_str.split('#') if tag.strip()]
                hashtags.extend(tags)
    
        features['top_hashtags'] = dict(Counter(hashtags).most_common(20))
    
    return features

def extract_topics(df, num_topics=5):
    """
    Extract topics from text using LDA.
    
    Args:
        df (pandas.DataFrame): Dataset with processed text
        num_topics (int): Number of topics to extract
        
    Returns:
        tuple: (vectorizer, lda_model, feature_names, topics)
    """
    # Create a document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(df['processed_text'].dropna())
    
    # Apply LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract topics
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[f"Topic {topic_idx+1}"] = top_words
    
    return vectorizer, lda, feature_names, topics

def extract_time_patterns(df):
    """
    Extract temporal patterns from the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset with timestamp information
        
    Returns:
        dict: Dictionary of temporal patterns
    """
    time_patterns = {}
    
    # Check if time columns exist
    if 'Hour' in df.columns:
        # Hour of day distribution
        hourly_counts = df['Hour'].value_counts().sort_index()
        time_patterns['hourly_distribution'] = hourly_counts.to_dict()
        time_patterns['peak_hour'] = hourly_counts.idxmax()
        
        # Define time periods
        morning = range(5, 12)  # 5 AM to 11 AM
        afternoon = range(12, 17)  # 12 PM to 4 PM
        evening = range(17, 22)  # 5 PM to 9 PM
        night = list(range(22, 24)) + list(range(0, 5))  # 10 PM to 4 AM
        
        # Count posts by time period
        time_periods = {
            'Morning': df[df['Hour'].isin(morning)].shape[0],
            'Afternoon': df[df['Hour'].isin(afternoon)].shape[0],
            'Evening': df[df['Hour'].isin(evening)].shape[0],
            'Night': df[df['Hour'].isin(night)].shape[0]
        }
        time_patterns['time_periods'] = time_periods
        
    # Day of week patterns if available
    if 'Timestamp' in df.columns:
        df['DayOfWeek'] = df['Timestamp'].dt.day_name()
        day_counts = df['DayOfWeek'].value_counts()
        time_patterns['day_of_week'] = day_counts.to_dict()
        time_patterns['most_active_day'] = day_counts.idxmax()
    
    return time_patterns
