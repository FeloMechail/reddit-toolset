import streamlit as st
import pandas as pd
from search_reddit import search


posts_df = None
comments_df = None

st.set_page_config(page_title="Query Reddit")

#st.sidebar.success("Select tools")

def main_page():
    st.title("Query Reddit")
    queries = st.text_area("Enter a query")
    queries = queries.replace(" ", "")
    #create a list and remove empty strings
    queries = queries.split("\n")
    queries = list(filter(None, queries))
    st.write("Your queries are: ")
    st.write(queries)
    subreddit = st.text_area("Enter a subreddits")
    subreddit = subreddit.replace(" ", "")
    #create a list and remove empty strings
    subreddit = subreddit.split("\n")
    subreddit = list(filter(None, subreddit))
    st.write("Your subreddits are: ")
    st.write(subreddit)

    num = st.number_input("Enter number of posts to search", min_value=1, max_value=100, value=50)

    #button to start the search
    if st.button("Search"):

        if st.button("cancel"):
            st.stop()
            
        #if the user has not entered any queries or subreddits
        if len(queries) == 0 or len(subreddit) == 0:
            st.error("Please enter at least one query and one subreddit")
        else:
            #call the function to search for the queries in the subreddits
            with st.spinner('Searching...'):
                global posts_df
                global comments_df
                posts_df, comments_df = search(queries, subreddit, num)
    
            if posts_df is None:
                st.error("No results found")
    else:
        st.stop()



if __name__ == "__main__":
    main_page()
    if posts_df is not None:
        st.markdown(f"Found <span style='color:green'>{len(posts_df)}</span> posts", unsafe_allow_html=True)
        

