#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import streamlit as st
pred_model_shrink = pd.read_csv('C:/Users/benja/OneDrive/Documents/Ironhack/PROJECTS/MUBI/data.csv', index_col=False)


# In[2]:


def find_similar_movies(pred_model_shrink):
    # User input
    input_title = st.text_input("Please enter a movie that you like:")

    if st.button('Find Similar Movies'):
        if input_title:
            # Step 1: Check for title similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(pred_model_shrink['Titles'])
            input_tfidf = vectorizer.transform([input_title])
            similarities = cosine_similarity(input_tfidf, tfidf_matrix)

            # Find the index of the movie title that is most similar to the input
            similar_index = similarities.argmax()

            # Check if similarity is above 90%
            if similarities[0][similar_index] < 0.9:
                return "Sorry, I don't know that movie at the moment."

            # Step 2: Filter DataFrame
            input_movie_features = pred_model_shrink.iloc[similar_index, 1:]
            columns_with_ones = input_movie_features[input_movie_features == 1].index
            filtered_df = pred_model_shrink.loc[(pred_model_shrink[columns_with_ones] == 1).any(axis=1)]

            # If the filtered DataFrame is empty, return the message
            if filtered_df.empty:
                return "I am sorry I can't help at the moment. Try with another movie."

            # Step 3: Run suggestion model
            filtered_df['sum_of_features'] = filtered_df.iloc[:, 1:].sum(axis=1)
            sorted_df = filtered_df.sort_values(by='sum_of_features', ascending=False)

            # Step 4: Return results
            top_movies = sorted_df.head(3)['Titles'].tolist()
            return top_movies
        else:
            return "Please enter a movie title to get suggestions."

# Example usage:
st.title('Movie Suggester App')
suggestions = find_similar_movies(pred_model_shrink)
if isinstance(suggestions, list):
    st.write('For sure you will like:')
    for movie in suggestions:
        st.write(movie)
else:
    st.write(suggestions)

