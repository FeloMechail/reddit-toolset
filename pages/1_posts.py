import pandas as pd
import streamlit as st
import os
import sys

sys.path.append('../')
from analysis import sentiment_analysis, topic_modelling

st.set_page_config(page_title="posts")

st.sidebar.header("Databses of posts")
st.sidebar.markdown("Select a database to view")


def view_db(db_file):
    df = pd.read_csv(f"db/{db_file}")
    comments_df = pd.read_csv(f"db/{db_file.split('.')[0]}_comments.csv")
    
    unique_subreddits = df['subreddit'].unique()

    len_subreddits = [len(df[df['subreddit'] == subreddit]) for subreddit in unique_subreddits]

    options = [f"{title}\t{length} " for title, length in zip(unique_subreddits, len_subreddits)]

    subreddit = st.sidebar.selectbox("Select Subreddit", options)

    subreddit = subreddit.split('\t')[0]


    # Display posts for selected subreddit
    st.header(f"Posts in r/{subreddit}")

    filtered_posts = df[df['subreddit'] == subreddit]
    post = st.selectbox('Select Post', filtered_posts['title'])

    post_info = filtered_posts[filtered_posts['title'] == post]
    
    #body
    st.subheader(post_info['title'].values[0])
    st.write(post_info['body'].values[0])
    st.markdown(f"#### Score: {post_info['score'].values[0]} | Author: {post_info['author'].values[0]} | comments: {post_info['comms_num'].values[0]}")

    try:
        st.info(f"Topics: {post_info['topic'].values[0]}")
    except:
        st.info("Topics: Not analysed")

    #comments
    st.markdown('## Comments')
    st.markdown('---')

    filtered_comments = comments_df[comments_df['submission_id'] == post_info['id'].values[0]]

    for index, comment in filtered_comments.iterrows():
        try:
            st.markdown(f"##### u/{comment['name']} | {comment['sentiment']} | {format(comment['score'], '.2%')}")
            st.markdown(f"{comment['comment_body']}")
            st.markdown("---")
        except:
            st.markdown(f"##### u/{comment['name']}")
            st.markdown(f"{comment['comment_body']}")
            st.markdown("---")

def main():
    #read all csv files in db folder
    with st.spinner('Loading db...'):
        db_files = os.listdir("db")
        db_files = [file for file in db_files if file.endswith(".csv")]

        queries = [file for file in db_files if "comment" not in file]

        selected_db = st.sidebar.selectbox("Select a database", queries)
        comments = [file for file in db_files if "comment" in file]
        
        if selected_db:
            view_db(selected_db)

    #column with 2 buttons
    col1, col2 = st.sidebar.columns(2)

    #col1 with key refresh
    if col1.button("Refresh"):
        main()

    if col2.button("Analyse"):
        with st.spinner('Analysing...'):
            sentiment_analysis(selected_db)
            topic_modelling(selected_db)
            if st.button("cancel"):
                st.stop()
        st.success("Done")
        main()

        
        




if __name__ == "__main__":
    main()

