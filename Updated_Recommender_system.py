
import pandas as pd
import numpy as np

credits = pd.read_csv("tmdb_5000_credits.csv")
movies_df = pd.read_csv("tmdb_5000_movies.csv")

credits.head()
movies_df.head()

print("Credits:",credits.shape)
print("Movies Dataframe:",movies_df.shape)

#Since in both Dataframes, we have same id column named, 'id' and 'movie_id'
credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})

#Now we'll merge both dataframes!
movies_df_merge = movies_df.merge(credits_column_renamed, on='id')
movies_df_merge.head()

movies_df_merge.shape

#We'll drop these columns since, it doesnt have any much use for recommendation systems!
movies_cleaned_df = movies_df_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
movies_cleaned_df.head()

movies_cleaned_df.info()


# ## Content Based Recommendation System
# 
# #### Now lets make a recommendations based on the movie’s plot summaries given in the overview column. So if our user gives us a movie title, our goal is to recommend movies that share similar plot summaries.
# 
# #### Now in this, in order to use cosine similarity in content based recommendation system, we need to represent the movies in form of matrix


movies_cleaned_df.head(2)['overview']

#Here we'll convert our paragraph into documentary matrix
from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')

# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'])

tfv_matrix
tfv_matrix.shape

#Now here, each of the summary vector in represented in 0/1, with the help of sigmid_kernel
from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
#Here, most imp!!!, each summary vector will be compared with other summary vector, eg.summ1-summ1=1,summ1 with summ2=0.75
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

#Basically here, sig[0], we'll represent, how summary vector 1,is related to summ2,summ3,summ4....summ'n'!!!
sig[0]


# Reverse mapping of indices and movie titles
indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()

indices['Newlyweds']
sig[47]

#Here, well take the movie we want recommendations, and using "enumerate", this will provide index for all recommended movies!
list(enumerate(sig[indices['Newlyweds']]))


#Arranging it in ascending order!
sorted(list(enumerate(sig[indices['Newlyweds']])), key=lambda x: x[1], reverse=True)
 ### So, based on above Code, we'll create a function, which takes the indices,takes the sig[value]
# ### Then, arranging those reccomendation in ascending, selecting the top 10 index of recommended movies,
# ### And, then getting orignal movie titles, based on the movie index!!!!

def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_cleaned_df['original_title'].iloc[movie_indices]

# Testing our content-based recommendation system with the seminal film Spy Kids
give_rec('Avatar')
