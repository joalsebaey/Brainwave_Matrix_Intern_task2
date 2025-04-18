"""
Social Media Sentiment Analysis with NLP
----------------------------------------
This script performs comprehensive sentiment analysis on social media data using
Natural Language Processing techniques. It includes:
1. Data loading and preprocessing
2. Text cleaning and normalization
3. Sentiment analysis using multiple approaches
4. Feature extraction (word frequencies, n-grams)
5. Visualization of sentiment trends and insights
"""

import os
import argparse
from data_processing.data_loader import load_data
from data_processing.preprocessor import preprocess_text
from analysis.sentiment_analysis import extract_sentiment_textblob, extract_sentiment_from_dataset
from analysis.feature_extraction import extract_features, extract_topics
from visualization.plot_functions import (
    plot_sentiment_distribution,
    plot_sentiment_trend,
    plot_top_sentiment_engagement,
    plot_sentiment_by_platform,
    plot_top_hashtags,
    plot_activity_by_hour,
    plot_sentiment_by_country,
    generate_wordcloud
)

# Create output directory for visualizations
OUTPUT_DIR = 'visualization_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main(file_path):
    """
    Main function to execute the analysis pipeline.
    
    Args:
        file_path (str): Path to the dataset
    """
    # Load and prepare data
    df = load_data(file_path)
    
    if df is None:
        print("Error: Failed to load dataset.")
        return
    
    print("\nPerforming sentiment analysis and generating visualizations...\n")
    
    # Apply text preprocessing
    df['tokens'], df['processed_text'] = zip(*df['Text'].apply(preprocess_text))
    
    # Extract sentiment using TextBlob (in addition to existing sentiment)
    df['sentiment_score'], df['computed_sentiment'] = zip(*df['processed_text'].apply(extract_sentiment_textblob))
    
    # Alternative: Use existing sentiment categories
    df['numeric_sentiment'] = df.apply(extract_sentiment_from_dataset, axis=1)
    
    # Extract text features
    features = extract_features(df)
    
    # Generate visualizations
    plot_sentiment_distribution(df, OUTPUT_DIR)
    plot_sentiment_trend(df, OUTPUT_DIR)
    plot_top_sentiment_engagement(df, OUTPUT_DIR)
    plot_sentiment_by_platform(df, OUTPUT_DIR)
    plot_top_hashtags(df, OUTPUT_DIR)
    plot_activity_by_hour(df, OUTPUT_DIR)
    plot_sentiment_by_country(df, OUTPUT_DIR)
    
    # Generate word cloud if WordCloud is available
    try:
        generate_wordcloud(df, OUTPUT_DIR)
    except Exception as e:
        print(f"Note: WordCloud visualization skipped: {e}")
    
    print("\nAnalysis complete! Generated visualizations:")
    print("1. Sentiment Distribution in Social Media Posts")
    print("2. Sentiment Trends Over Time")
    print("3. Average Engagement by Sentiment")
    print("4. Sentiment Analysis by Social Media Platform")
    print("5. Top 10 Hashtags in Social Media Posts")
    print("6. Posting Activity by Hour of Day")
    print("7. Sentiment Distribution Across Top 8 Countries")
    
    # Optional: Try to extract topics
    try:
        vectorizer, lda, feature_names, topics = extract_topics(df)
        print("8. Topic Modeling Results")
        for topic, words in topics.items():
            print(f"   - {topic}: {', '.join(words)}")
    except Exception as e:
        print(f"Note: Topic modeling skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Social Media Sentiment Analysis')
    parser.add_argument('--data_path', type=str, default="data/raw/socialmediadataset.csv",
                        help='Path to the social media dataset CSV file')
    args = parser.parse_args()
    
    main(args.data_path)
