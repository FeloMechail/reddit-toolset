from enum import unique
from itertools import count
from matplotlib import category
import streamlit as st
import pandas as pd
import nltk 
import re
import string
import spacy
from collections import Counter
from transformers import pipeline
from spacy import displacy
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from keybert import KeyBERT
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import wordcloud





#using two dataframes, provinces and comments, create a relationship between the two and a reddit like interface to display the data with body and comments in a tree like structure    

#pre processing
classifier = pipeline("sentiment-analysis", model="sbcBI/sentiment_analysis_model", top_k=None, device=0)
nlp = spacy.load('en_core_web_sm')
kw_model = KeyBERT(model="BAAI/bge-small-en-v1.5")


# provinces_df = pd.read_csv('provinces.csv')
# comments_df = pd.read_csv('comments.csv')

# body = provinces_df['body']
# comments = comments_df['comment_body']



def preprocessing(df):
    #lowercase
    df = df.str.lower()

    #remove punctuation
    df = df.str.replace('[^\w\s]','')

    #remove stopwords
    stop_words = nltk.corpus.stopwords.words('english')
    df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    #remove frequent words
    freq = pd.Series(' '.join(df).split()).value_counts()[:10]
    freq = list(freq.index)
    df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in freq]))

    #remove rare words
    rare = pd.Series(' '.join(df).split()).value_counts()[-10:]
    rare = list(rare.index)
    df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in rare]))

    #stemming
    ps = nltk.PorterStemmer()
    df = df.apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))

    #lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    df = df.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    
    #remove emojis
    df = df.str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
    df = df.str.replace('[^\x00-\x7F]+', '', flags=re.UNICODE)

    #remove urls
    df = df.str.replace('http\S+|www.\S+', '', case=False)

    #remove html tags
    df = df.str.replace('<.*?>', '', case=False)

    #remove numbers
    df = df.str.replace('\d+', '', case=False)

    return df



def sentiment_analysis(selected_db):

    sentiment_df = {
    "comment_id": [],
    "sentiment": [],
    "score": []
        }
    
    with st.spinner('Preprocessing...'):
        comments = pd.read_csv(f"db/{selected_db.split('.')[0]}_comments.csv")

        #check if sentiment analysis has already been done
        if 'sentiment' in comments.columns:
            print('sentiment already done')
            return comments
       
        #print(comments)

        #replace comments_body with preprocessed comments
        #comments['PrePro_comments'] = preprocessing(comments['comment_body'])


            #sentiment analysis
        for index, comment in comments.iterrows():
            if len(comment['comment_body']) <= 260:
                sentiment = classifier(comment['comment_body'])
                sentiment_df['comment_id'].append(comment['comment_id'])
                highest_score = max(sentiment[0], key=lambda x:x['score']) # type: ignore
                highest_label = highest_score['label']
                
                if highest_label == 'LABEL_0': highest_label = 'NEG'
                elif highest_label == 'LABEL_2': highest_label = 'POS'
                else: highest_label = 'NEU'

                hgihtest_score = highest_score['score']
                sentiment_df['sentiment'].append(highest_label)
                sentiment_df['score'].append(hgihtest_score)

        sentiment_df = pd.DataFrame(sentiment_df)
        comments_df = comments.merge(sentiment_df, on='comment_id')
        comments_df.to_csv(f"db/{selected_db.split('.')[0]}_comments.csv", index=False)
    
    return comments_df




def topic_modelling(selected_db):
    body = pd.read_csv(f"db/{selected_db}")
    
    body['body'] = body['body'].dropna()
    body['body'] = body['body'].astype(str)

    # Check if topic modelling has already been done
    if 'topic' in body.columns:
        print('Topic modelling already done')
        return body
    
    topics_df = {
        "topic": [],
        "id": [],
    }   


    for index, post in body.iterrows():
        doc_embeddings, word_embeddings = kw_model.extract_embeddings(post['body'], stop_words="english", keyphrase_ngram_range=(1,2))
        keywords = kw_model.extract_keywords(post['body'], stop_words="english", keyphrase_ngram_range=(1,2),doc_embeddings=doc_embeddings, word_embeddings=word_embeddings,highlight=True, use_mmr=True)
        keywords = [keyword[0] for keyword in keywords]

        topics_df['topic'].append(keywords)
        topics_df['id'].append(post['id'])

        print(keywords)

    topics_df = pd.DataFrame(topics_df)

    body = body.merge(topics_df, on='id')

    body.to_csv(f"db/{selected_db}", index=False)

    return body


def top_topics_count(selected_db):
    topics = pd.read_csv(f"db/{selected_db}")
    sentiment = pd.read_csv(f"db/{selected_db.split('.')[0]}_comments.csv")

    #print(sentiment.head())

    if 'topic' not in topics.columns:
        print('topic modelling not done')
        return
    
    topics_df = {
        "Topics": [],
        "Post_id": [],
        "Sentiment": []
    }


    for index, post in topics.iterrows():
        topicss = post['topic'].strip('][').split(', ')
        #give me sentiment of comments with post id
        sentiments = sentiment[sentiment['submission_id'] == post['id']]
        sentiments = sentiments['sentiment'].value_counts().to_dict()
        #print(sentiment)

        topics_df['Topics'].append(topicss)
        topics_df['Post_id'].append(post['id'])
        topics_df['Sentiment'].append(sentiments)

    topics = pd.DataFrame(topics_df)

    topics.to_csv(f"db/{selected_db.split('.')[0]}_topics.csv", index=False)

    return topics


#samples by topic

def samples_by_topic(selected_db):
    topics = pd.read_csv(f"db/{selected_db.split('.')[0]}.csv") 

    topics_list = topics['Topics'].to_list()
    topics_list = [topic.strip('][').split(', ') for topic in topics_list]
    topics_list = [item.strip("\"") for sublist in topics_list for item in sublist]
    topics_list = [topic.strip("\'") for topic in topics_list]
    topics_list = [topic for topic in topics_list if topic != 'nan']

    topics_count = Counter(topics_list)
    topics_count_df = pd.DataFrame(topics_count.items(), columns=['Topic', 'Count'])

    topics_count_df.replace('heat pumps', 'heat pump', inplace=True)

    topics_count_df = topics_count_df.groupby(['Topic']).sum().reset_index()

    topics_count_df = topics_count_df.sort_values(by=['Count'], ascending=False)

    st.dataframe(topics_count_df)

    #-----------------------------

    st.markdown('## Top 10 Topics')
    #horizontal bar chart for top 10 topics
    fig, ax = plt.subplots()
    top10_topics = topics_count_df.head(10).sort_values(by=['Count'], ascending=True)
    ax.barh(top10_topics['Topic'], top10_topics['Count'])
    ax.set_xlabel('Count')
    ax.set_ylabel('Topic')
    ax.set_title('Top 10 Topics')
    st.pyplot(fig)

    #-----------------------------
    st.markdown('## Word Cloud')
    wordcloud = WordCloud(background_color='white').generate(' '.join(topics_list))
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    #-----------------------------

    st.markdown('## Sentiment by Topic')
    sentiment_list = topics['Sentiment'].to_list()
    sentiment_list = [eval(sentiment) for sentiment in sentiment_list]
    
    sentiment_df = pd.DataFrame(sentiment_list)
    sentiment_df = sentiment_df.fillna(0)
    sentiment_df = sentiment_df.astype(int)
    
    #create sentiment count bar graph from sentiment dict column
    fig, ax = plt.subplots()
    sentiment_df_sum = sentiment_df.sum().reset_index()
    sentiment_df_sum.columns = ['Sentiment', 'Count']
    ax.bar(sentiment_df_sum['Sentiment'], sentiment_df_sum['Count'])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment by Topic')
    st.pyplot(fig)


    #-----------------------------
    #sentiment by topic graph
    st.markdown('## Sentiment by Topic')
    topics['Topics'] = topics['Topics'].apply(lambda x: x.strip('][').split(', '))
    topics['Topics'] = topics['Topics'].apply(lambda x: [item.strip("\"") for item in x])
    topics['Topics'] = topics['Topics'].apply(lambda x: [item.strip("\'") for item in x])
    
    expanded_topics = {"Topics": [], "Sentiment": []}
    
    for index, row in topics.iterrows():
        topics = row['Topics']
        sentiment = row['Sentiment']

        for topic in topics:
            expanded_topics['Topics'].append(topic)
            expanded_topics['Sentiment'].append(sentiment)

    expanded_topics = pd.DataFrame(expanded_topics)

    expanded_topics.replace('heat pumps', 'heat pump', inplace=True)

    #eval sentiment dict column
    expanded_topics['Sentiment'] = expanded_topics['Sentiment'].apply(lambda x: eval(x))

    expanded_topics['Positive'] = expanded_topics['Sentiment'].apply(lambda x: get_sentiment_category(x, 'POS'))
    expanded_topics['Negative'] = expanded_topics['Sentiment'].apply(lambda x: get_sentiment_category(x, 'NEG'))
    expanded_topics['Neutral'] = expanded_topics['Sentiment'].apply(lambda x: get_sentiment_category(x, 'NEU'))
    
    #drop sentiment column
    expanded_topics = expanded_topics.drop(columns=['Sentiment'])
    expanded__topics = expanded_topics


    #group and add topic count
    topics_count = Counter(expanded_topics['Topics'].to_list())
    expanded_topics = expanded_topics.groupby(['Topics']).sum().reset_index()
    
    #add topic count column
    expanded_topics['Count'] = expanded_topics['Topics'].apply(lambda x: topics_count[x])

    #remove nan row from topics
    expanded_topics = expanded_topics[expanded_topics['Topics'] != 'nan']

    st.dataframe(expanded_topics)

    #-----------------------------
    #sentiment by topic graph, topic on y axis and sentiment % on x axis
    choice = st.selectbox('sort by', ['Positive', 'Negative', 'Neutral'])
    top10_topics = expanded_topics.sort_values(by=[choice], ascending=False).head(10)
    print(top10_topics)
    fig, ax = plt.subplots()
    ax.barh(top10_topics['Topics'], top10_topics['Positive'], label='Positive')
    ax.barh(top10_topics['Topics'], top10_topics['Negative'], left=top10_topics['Positive'], label='Negative')
    ax.barh(top10_topics['Topics'], top10_topics['Neutral'], left=top10_topics['Positive']+top10_topics['Negative'], label='Neutral')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Topic')
    ax.set_title('Sentiment by Topic')
    ax.legend()
    st.pyplot(fig)


    #-----------------------------
    #cluster topics
    st.markdown('## Clustering')
    clusters_df = cluster(expanded_topics['Topics'].to_list())
    #merge positive, negative and neutral columns to clusters_df
    clusters_df = clusters_df.merge(expanded_topics, on='Topics')
    clusters_df_copy = clusters_df.copy()
    st.dataframe(clusters_df)

    #sentiment by cluster graph
    st.markdown('## Sentiment by Cluster')
    clusters_df = clusters_df.drop(columns=['Topics'])
    clusters_df = clusters_df.groupby(['Clusters']).sum().reset_index()
    #drop large clusters
    clusters_df = clusters_df[clusters_df['Count'] < 100] 
    #choice1 = st.selectbox('sort by', ['Positive', 'Negative', 'Neutral'])
    clusters_df = clusters_df.sort_values(by=[choice], ascending=False)
    fig, ax = plt.subplots()
    ax.bar(clusters_df['Clusters'], clusters_df['Positive'], label='Positive')
    ax.bar(clusters_df['Clusters'], clusters_df['Negative'], bottom=clusters_df['Positive'], label='Negative')
    ax.bar(clusters_df['Clusters'], clusters_df['Neutral'], bottom=clusters_df['Positive']+clusters_df['Negative'], label='Neutral')
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Sentiment')
    ax.set_title('Sentiment by Cluster')
    ax.legend()
    st.pyplot(fig)

    #-----------------------------
    #new dataframe with clusters as index and topics as columns
    clusters_dff = pd.DataFrame()

    for i in range(0, 12):
        clusters_dff[f"cluster {i}"] = clusters_df_copy.groupby(['Clusters']).get_group(i)['Topics']

    st.dataframe(clusters_dff)
    
def get_sentiment_category(sentiment_dict, category):
    return sentiment_dict.get(category, 0)



def cluster(topics):
    # Convert topics to TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(topics)

    # Use KMeans to cluster the topics
    num_clusters = 12  # Adjust based on the desired number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # Assign clusters to topics
    clusters = kmeans.labels_

    # Print topics and their corresponding clusters
    for i in range(num_clusters):
        print(f"Cluster {i + 1}:")
        for j, topic in enumerate(topics):
            if clusters[j] == i:
                print(f"  - {topic}")

    #save clusters and topics to csv
    clusters_df = pd.DataFrame({'Topics': topics, 'Clusters': clusters})

    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    #show clusters in streamlit
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('Clusters')
    st.pyplot(fig)



    return clusters_df
    
    






