import streamlit as st
import kagglehub as kgh
import numpy as np
import pandas as pd
import math,copy
from sklearn.preprocessing import StandardScaler,MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

dataset_fil=pd.read_csv('imdb_top_1000.csv')
#replaced nan values with mean of the column
mean = np.nanmean(dataset_fil['Meta_score'])
dataset_fil['Meta_score'] = np.where(dataset_fil['Meta_score'].isna(), mean, dataset_fil['Meta_score'])
# Replaced empty strings with nan
dataset_fil['Gross'] = dataset_fil['Gross'].replace('', np.nan)

# Removed commas and converted it to float dtype
dataset_fil['Gross'] = dataset_fil['Gross'].str.replace(',', '', regex=True).astype(float)

# replaced nan with mean of col 
dataset_fil['Gross'].fillna(dataset_fil['Gross'].mean(), inplace=True)

numeric_features = ['IMDB_Rating','Meta_score','Gross']
# Scaling numeric features
scaler = StandardScaler()
X_numeric = scaler.fit_transform(dataset_fil[numeric_features])

genres_split = dataset_fil['Genre'].str.split('|')

# Multi-hot encoding
mlb = MultiLabelBinarizer()
X_genre = mlb.fit_transform(genres_split)

# Combine numeric and genre features
X_features = np.hstack([X_numeric, X_genre])
similarity_matrix = cosine_similarity(X_features)

def recommend_movie(movie_title, top_n=5):
    idx = dataset_fil[dataset_fil['Series_Title']==movie_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies_idx = [i[0] for i in sim_scores[1:top_n+1]]  
    return dataset_fil[['Series_Title','IMDB_Rating','Gross','Meta_score','Poster_Link']].iloc[top_movies_idx]




st.title("🎬 Movie Recommender System")

# Movie selection
selected_movie = st.selectbox("Enter movie name:", dataset_fil['Series_Title'])

# --- Display selected movie details ---
movie = dataset_fil[dataset_fil['Series_Title'] == selected_movie].iloc[0]

st.subheader(f"Selected Movie: {movie['Series_Title']}")
st.image(movie['Poster_Link'], width=100)  # small poster
st.write(f"**IMDB Rating:** {movie['IMDB_Rating']}")
st.write(f"**Meta Score:** {movie['Meta_score']}")
st.write(f"**Genre:** {movie['Genre']}")
st.write(f"**Cast:** {movie['Star1']},{movie['Star2']},{movie['Star3']},{movie['Star4']}")
st.write(f"**Description:** {movie.get('Overview', 'No description available')}")  # optional column

st.markdown("---")

# --- Display recommended movies ---
st.subheader("Recommended Movies:")

recommended = recommend_movie(selected_movie, top_n=5)

# Display recommendations horizontally with small posters
cols = st.columns(5)  # 5 movies per row
for i, row in recommended.iterrows():
    col = cols[i % 5]  # cycle through 5 columns
    with col:
        st.image(row['Poster_Link'], width=100)
        st.write(row['Series_Title'])

