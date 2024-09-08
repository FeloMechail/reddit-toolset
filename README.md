# Reddit Data Scraper and Analyzer
## Project Overview
This project is a Python-based application using streamlit designed to scrape posts and comments from specified subreddits using Reddit's API. The collected data is then processed and analyzed using various techniques, including the integration of language models and embeddings for enhanced insights.

## Features
* Reddit API Integration: Securely connects to Reddit's API using credentials stored in environment variables.
* Data Scraping: Collects detailed information about posts and comments from specified subreddits.
* Data Preprocessing: Cleans the text data by performing operations like lowercasing, removing punctuation, stopwords, frequent words, rare words, emojis, URLs, HTML tags, and numbers, and applying stemming and lemmatization.
* Topic Modeling: Uses the KeyBERT model, which leverages BERT embeddings, to extract keywords from posts and identify topics.
* Sentiment Scoring: Used sbcBI/Sentiment_analysis_model to perform sentiment analysis on the preprocessed text and assign scores to each post and comment
* Data Storage: Stores the scraped and processed data in CSV files for further analysis.
* Data Analysis: Created visual representations (e.g., graphs, charts, word cloud) to understand the overall mood within the subreddit

## Demo
![alt](demo.gif)
