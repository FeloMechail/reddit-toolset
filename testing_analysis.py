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
from transformers import pipeline

nlp = spacy.load('en_core_web_sm')

comments = pd.read_csv('db/ontario_comments.csv')
body = pd.read_csv('db/ontario.csv')

comments = comments['comment_body']
body = body['body']

analyzed_comments = pd.DataFrame(columns=['comment_body', 'sentiment', 'score'])
analyzed_body = pd.DataFrame(columns=['body', 'topics'])

#gpu
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier", top_k=None, device=0)


def topic_modelling(body):
    texts, article = [], []

    # for post in body:
    #     print(post)

    post = body[0]
    post = post + " New York"
    
    #remove urls
    post = re.sub(r'http\S+', '', post)

    post = list(filter(bool, post.splitlines()))
    
    for paragraph in post:
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

    print(texts)
    
    #dictionary
    dictionary = Dictionary(texts)
    article = [dictionary.doc2bow(text) for text in texts]
    
    #lda model
    lda = LdaModel(corpus=article, id2word=dictionary, num_topics=1, passes=10)
    
    #get top 3 topics
    topics = lda.print_topics(num_words=3)


    print(topics)
    

#topic_modelling(body)
    
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

def sentiment_analysis(comments):

    comments_with_preprocessing = preprocessing(comments)
    comments_with_preprocessing = comments_with_preprocessing.tolist()

    classified = classifier(comments_with_preprocessing)

    comments_sent = {
    "comment_id": [],
    "sentiment": [],
    "score": []
        }
    print(classified)
    
    for classi in classified:
        print(classi)
        comments_sent['comment_id'].append(classi[0]['label'])
        comments_sent['sentiment'].append(classi[0]['score'])
        comments_sent['score'].append(classi[0]['score'])
    
    comments_sent = pd.DataFrame(comments_sent)
    #save to csv
    comments_sent.to_csv('comments_sent.csv', index=False)


topic_modelling(body)
