import streamlit as st
import pandas as pd
import os
from analysis import *


st.set_page_config(page_title="Graphs")

st.sidebar.header("View Graphs")
st.sidebar.markdown("Select a database to view")


def view_db(db_file):
    df = pd.read_csv(f"db/{db_file}")
    comments_df = pd.read_csv(f"db/{db_file.split('topics.')[0]}comments.csv")
    
    st.header(f"Topics in {db_file.split('_topics.')[0]}")
    st.markdown('---')
    #view df
    st.dataframe(df)


def main():
    with st.spinner('Loading db...'):
        db_files = os.listdir("db")
        db_files = [file for file in db_files if file.endswith("topics.csv")]

        queries = [file for file in db_files if "comment" not in file]

        selected_db = st.sidebar.selectbox("Select a topic database", queries)
        comments = [file for file in db_files if "comment" in file]
        
        if selected_db:
            view_db(selected_db)
            samples = samples_by_topic(selected_db)


if __name__ == "__main__":
    main()