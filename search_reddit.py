import praw as pr
import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st


def reddit_instance():
    
    load_dotenv()
    
    reddit = pr.Reddit(
        client_id = os.environ.get('CLIENTID'),
        client_secret = os.environ.get('CLIENTSECRET'),
        user_agent = os.environ.get('USERAGENT')
    )

    return reddit   

def search(queries, subreddits, num):

    posts_df, comments_df = None, None

    print("Searching...")

    #if csv files already exist, return them
    if os.path.isfile(f"{'+'.join(subreddits)}.csv") and os.path.isfile(f"{'+'.join(subreddits)}_comments.csv"):
        posts_df = pd.read_csv(f"{'+'.join(subreddits)}.csv")
        comments_df = pd.read_csv(f"{'+'.join(subreddits)}_comments.csv")
        return posts_df, comments_df

    reddit = reddit_instance()

    posts = {
        "title": [],
        "author": [], 
        "score": [],
        "id": [],
        "url": [],
        "comms_num": [],
        "created": [],
        "subreddit": [],
        "body": [],
        "permalink": []
    }

    comments = {
        "name": [],
        "submission_id": [],
        "comment_id": [],
        "comment_parent_id": [],
        "comment_body": [],
        "permalink": []
    }
    try:
        for submissions in reddit.subreddit("+".join(subreddits)).search("+".join(queries), limit=num, sort="top", time_filter="all"):
            posts["title"].append(submissions.title)
            posts["score"].append(submissions.score)
            posts["id"].append(submissions.id)
            posts["url"].append(submissions.url)
            posts["comms_num"].append(submissions.num_comments)
            posts["created"].append(submissions.created)
            posts["subreddit"].append(submissions.subreddit)
            posts["body"].append(submissions.selftext)
            posts["permalink"].append(submissions.permalink)
            posts["author"].append(submissions.author)

            #get only parent comments
            submissions.comments.replace_more(limit=0)
            for comment in submissions.comments:
                comments["name"].append(comment.author)
                comments["submission_id"].append(comment.submission.id)
                comments["comment_id"].append(comment.id)
                comments["comment_parent_id"].append(comment.parent_id)
                comments["comment_body"].append(comment.body)
                comments["permalink"].append(comment.permalink)

    except Exception as e:
        st.error(e)
        st.stop()

    if len(posts["title"]) != 0:
        posts_df = pd.DataFrame(posts)
        comments_df = pd.DataFrame(comments)

        posts_df.to_csv(f"db\{'+'.join(subreddits)}.csv", index=False)
        comments_df.to_csv(f"db\{'+'.join(subreddits)}_comments.csv", index=False)

    return posts_df, comments_df