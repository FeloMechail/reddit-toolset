import streamlit as st
import pandas as pd
import nltk 
import re
import string
import spacy
from collections import Counter
from transformers import pipeline



#using two dataframes, provinces and comments, create a relationship between the two and a reddit like interface to display the data with body and comments in a tree like structure    

#pre processing
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
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

    return df

sentiment_analysis = {
    "comment_id": [],
    "sentiment": [],
    "score": []
}

def sentiment_analysis(comments_df):
    with st.spinner('Preprocessing...'):
        #replace comments_body with preprocessed comments
        comments_df['comment_body'] = preprocessing(comments_df['comment_body'])

            #sentiment analysis
        for index, comment in comments_df.iterrows():
            if len(comment['comment_body']) <= 260:
                sentiment = classifier(comment['comment_body'])
                sentiment_analysis['comment_id'].append(comment['comment_id'])
                highest_score = max(sentiment[0], key=lambda x:x['score'])
                highest_label = highest_score['label']
                hgihtest_score = highest_score['score']
                sentiment_analysis['sentiment'].append(highest_label)
                sentiment_analysis['score'].append(hgihtest_score)

        sentiment_df = pd.DataFrame(sentiment_analysis)
        comments_df = comments_df.merge(sentiment_df, on='comment_id')
    
    return comments_df






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













    
    






