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
from bertopic import BERTopic




#using two dataframes, provinces and comments, create a relationship between the two and a reddit like interface to display the data with body and comments in a tree like structure    

#pre processing
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier", top_k=None, device=0)
model = BERTopic(verbose=True, embedding_model="paraphrase-MiniLM-L6-v2", min_topic_size=7)
nlp = spacy.load('en_core_web_sm')

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
       
        print(comments)

        #replace comments_body with preprocessed comments
        #comments['PrePro_comments'] = preprocessing(comments['comment_body'])


            #sentiment analysis
        for index, comment in comments.iterrows():
            if len(comment['comment_body']) <= 260:
                sentiment = classifier(comment['comment_body'])
                sentiment_df['comment_id'].append(comment['comment_id'])
                highest_score = max(sentiment[0], key=lambda x:x['score'])
                highest_label = highest_score['label']
                hgihtest_score = highest_score['score']
                sentiment_df['sentiment'].append(highest_label)
                sentiment_df['score'].append(hgihtest_score)

        sentiment_df = pd.DataFrame(sentiment_df)
        comments_df = comments.merge(sentiment_df, on='comment_id')
        comments_df.to_csv(f"db/{selected_db.split('.')[0]}_comments.csv", index=False)
    
    return comments_df


def topic_modelling(selected_db):
    body = pd.read_csv(f"db/{selected_db}")

    # for post in body:
    #     print(post)

    topics_df = {
        "topic": [],
        "id": [],
    }

    for index, post in body.iterrows():
        #remove urls
        texts, article = [], []
        
        bodys = post['body']

        #remove urls
        #print(post)
        #to string
        bodys = str(bodys)
        
        bodys = bodys.replace('http\S+|www.\S+', '',)
        
        bodys = list(filter(bool, bodys.splitlines()))
        
        
        for paragraph in bodys:

            paragraph = nlp(paragraph)
            

            for word in paragraph:
                if not word.is_stop and not word.is_punct and not word.like_num and not word.like_url and word.text != "I" and not word.is_space:
                    article.append(word.lemma_)
            
            texts.append(article)
            article = []

        
        #bigrams to help with "new york" 
        bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        texts = [bigram_mod[doc] for doc in texts]

        #print(texts)
        
        #dictionary
        dictionary = Dictionary(texts)
        article = [dictionary.doc2bow(text) for text in texts]
        
        #lda model
        lda = LdaModel(corpus=article, id2word=dictionary, num_topics=1, passes=10)
        
        #get top 3 topics
        topics = lda.print_topics(num_words=5)

        topics_df['topic'].append(topics)
        topics_df['id'].append(post['id'])

    topics_df = pd.DataFrame(topics_df)
    print(topics_df)
    body = body.merge(topics_df, on='id')
    body.to_csv(f"db/{selected_db}", index=False)

    return body




# '''
# provinces = {
#     "title": [],
#     "score": [],
#     "id": [],
#     "url": [],
#     "comms_num": [],
#     "created": [],
#     "subreddit": [],
#     "body": [],
#     "permalink": []
# }
# '''

# '''
# comments = {
#     "name": [],
#     "submission_id": [],
#     "comment_id": [],
#     "comment_parent_id": [],
#     "comment_body": [],
#     "comment_link_id": []
# }
# '''



# subreddit = st.sidebar.selectbox("Select Subreddit", provinces_df['subreddit'].unique())

# # Display posts for selected subreddit
# st.header(f"Posts in r/{subreddit}")

# filtered_posts = provinces_df[provinces_df['subreddit'] == subreddit]
# post = st.selectbox('Select Province(s)', filtered_posts['title'])

# post_info = filtered_posts[filtered_posts['title'] == post]

# st.subheader(post_info['title'].values[0])
# st.write(f"Score: {post_info['score'].values[0]} | Comments: {post_info['comms_num'].values[0]} | Created: {post_info['created'].values[0]}")
# st.write(post_info['body'].values[0])

# # Display comments for selected post, and each comment in a box
# st.markdown('# Comments')
# st.markdown('---')

# filtered_comments = comments_df[comments_df['submission_id'] == post_info['id'].values[0]]

# for index, comment in filtered_comments.iterrows():
#     st.markdown(f"##### u/{comment['name']} | {comment['sentiment']} | {comment['score']}")
#     st.markdown(f"{comment['comment_body']}")
#     st.markdown("---")













    
    






