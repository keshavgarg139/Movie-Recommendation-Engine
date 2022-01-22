#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:10:06 2019

@author: Keshav
"""

import re
from statistics import mean

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

import pandas as pd
import numpy as np
import math
import time

from gensim import models,corpora
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity

import itertools as it

import pickle

print("modules are imported")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def ptime():
    return time.asctime(time.localtime(time.time()))
    

def remove_stopwords(sentence):
    stop = list(stopwords.words('english'))
    sent_split = list(y for y in sentence.split() if y not in stop)
    return ' '.join(sent_split)

def remove_special_chars(text):
    final = [re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in text.split("\n")]
    return " ".join(final)
    
def replace_special_chars(sent):
    chars = [',','.',':',';','-','/','(',')']
    for char in chars:
        sent = sent.replace(char,' ')
        
    return sent    

def stripHTMLTags (html):
    text = html
    rules = [
    { r'>\s+' : u'>'},                  # remove spaces after a tag opens or closes
    { r'\s+' : u' '},                   # replace consecutive spaces
    { r'\s*<br\s*/?>\s*' : u'\n'},      # newline after a <br>
    { r'</(div)\s*>\s*' : u'\n'},       # newline after </p> and </div> and <h1/>...
    { r'</(p|h\d)\s*>\s*' : u'\n\n'},   # newline after </p> and </div> and <h1/>...
    { r'<head>.*<\s*(/head|body)[^>]*>' : u'' },     # remove <head> to </head>
    { r'<a\s+href="([^"]+)"[^>]*>.*</a>' : r'\1' },  # show links instead of texts
    { r'[ \t]*<[^<]*?/?>' : u' ' },            # remove remaining tags
    { r'^\s+' : u'' }                   # remove spaces at the beginning
    ]
 
    for rule in rules:
        for (k,v) in rule.items():
            regex = re.compile (k)
            text  = regex.sub (v, text)
 
    special = {
    '&nbsp;' : ' ', '&amp;' : '&', '&quot;' : '"',
    '&lt;'   : '<', '&gt;'  : '>'
    }
 
    for (k,v) in special.items():
        text = text.replace (k, v)
        
    return text

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_sent(sent):
    lemmatizer = WordNetLemmatizer()
    new_sent = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sent)]
    return " ".join(new_sent)

def clean_sentence(sent):
    s1 = stripHTMLTags(sent.lower())
    s2 = lemmatize_sent(replace_special_chars(s1))
    s3 = remove_stopwords(s2)
    return s3

def get_lsi_model(all_corpus):
    
    Corp = list(x.split() for x in all_corpus)
    dictionary = corpora.Dictionary(Corp)
    train_bow_corpus = [dictionary.doc2bow(text) for text in Corp]
    lsi = models.LsiModel(train_bow_corpus,num_topics=250, onepass=False, power_iters=3, extra_samples=150)
    return lsi,dictionary

def get_similarity(model,dictionary,target_corpus,query):
    
    print(ptime(),"Calculating  the index")
    index_tmpfile = get_tmpfile('index')
    target_corp = list(x.split() for x in target_corpus)
    target_bow_corp = [dictionary.doc2bow(text) for text in target_corp]
    target_modelled_corpus = model[target_bow_corp]
    index = Similarity(index_tmpfile, target_modelled_corpus, num_features=len(dictionary))
    print(ptime(),"Done, Calculating  the index")
    
    query_vec = model[dictionary.doc2bow(query.lower().split())]
    
#    print(ptime(), "Vector lda for the query is :-  ",query_vec)
    similarity = list(index[query_vec])
    return similarity
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

folder_path = "M:/MBA/Sem 3/Big Data for Managers/Project/codes/"
data_path = "M:/MBA/Sem 3/Big Data for Managers/Project/codes/ml-20m/"
trained_data_path = "M:/MBA/Sem 3/Big Data for Managers/Project/codes/trained_model_data/"
output_folder = folder_path + "output/"

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
movies = pd.read_csv(data_path + 'movies.csv')

ratings = pd.read_csv(data_path + 'ratings.csv')

genome_scores = pd.read_csv(data_path + 'genome-scores.csv')

genome_tags = pd.read_csv(data_path + 'genome-tags.csv')

tags = pd.read_csv(data_path + 'tags.csv')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

genome_scores_filtered = genome_scores[genome_scores.relevance>0.15]

genome_scores_filtered_tags = pd.merge(genome_scores_filtered, genome_tags, on='tagId')

filtered_genome_tags_relev = pd.merge(genome_scores_filtered,genome_tags)

filtered_movieID_tags_relev = filtered_genome_tags_relev[~filtered_genome_tags_relev['movieId'].isin(tags['movieId'].tolist())]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

del genome_scores, genome_tags, genome_scores_filtered

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tag_genomescore_advanced = 2

if tag_genomescore_advanced == 1:
    movieId_tags_df = tags.loc[:,['movieId','tag']]
elif tag_genomescore_advanced == 2:
    movieId_tags_df = genome_scores_filtered_tags.loc[:,['movieId','tag']]    
elif tag_genomescore_advanced == 3:
    movieId_tags_df = filtered_movieID_tags_relev.loc[:,['movieId','tag']]

movieId_tags_df['tag'] = [str(x) for x in movieId_tags_df['tag']]

uncleaned_cleaned_tags_df = pd.DataFrame()

uncleaned_cleaned_tags_df['tag'] = list(set(movieId_tags_df['tag'].tolist()))
uncleaned_cleaned_tags_df['cleaned_tags'] = [clean_sentence(x) for x in uncleaned_cleaned_tags_df['tag'].tolist()]

movieId_tags_df_cleaned = pd.merge(movieId_tags_df,uncleaned_cleaned_tags_df, on='tag')

del movieId_tags_df_cleaned['tag']

movieId_tags_joined_df_cleaned = movieId_tags_df_cleaned.groupby('movieId')['cleaned_tags'].apply(','.join).reset_index()

with open(trained_data_path+'movieId_tags_joined_df_cleaned.pickle', 'wb') as f:
    pickle.dump(movieId_tags_joined_df_cleaned, f)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

with open(trained_data_path+'movieId_tags_joined_df_cleaned.pickle', 'rb') as f:
    movieId_tags_joined_df_cleaned = pickle.load(f)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

whole_tags_corpus_available = list(movieId_tags_joined_df_cleaned.cleaned_tags)
(lsi,dictionary) = get_lsi_model(all_corpus=whole_tags_corpus_available)

with open(trained_data_path+'lsi_model.pickle', 'wb') as f:
    pickle.dump(lsi, f)
with open(trained_data_path+'lsi_dictionary.pickle', 'wb') as f:
    pickle.dump(dictionary, f)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
with open(trained_data_path+'lsi_model.pickle', 'rb') as f:
    lsi = pickle.load(f) 
with open(trained_data_path+'lsi_dictionary.pickle', 'rb') as f:
    dictionary = pickle.load(f)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

unique_genres_or = list(set(movies['genres'].tolist()))
unique_genres = []
for x in unique_genres_or:
    unique_genres.append(x.split('|'))
    
unique_genres = sorted(list(set(list(it.chain.from_iterable(unique_genres)))))
unique_genres_ids = range(1,21)

genre_id_dic = dict(zip(unique_genres,unique_genres_ids))

genre_ids_movies = []

for x in list(movies['genres']):
    id_list = []
    for y in x.split('|'):
        id_list.append(genre_id_dic[y])
    genre_ids_movies.append(set(id_list))

movies['genre_id_list'] = genre_ids_movies   
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


start_time = "############### " + ptime()+" EVALUATION BEGINS ###############"

query_movie_id = 75548
query_row = movies[movies.movieId==query_movie_id]
query_genres = query_row.iloc[0,3]
query_movie_name = query_row.iloc[0,1]
print(query_genres)

threshold = math.ceil(len(query_genres)/2)+1 if len(query_genres)>=2 else 1

matches = []
for x in movies.genre_id_list:
    matches.append(len(x.intersection(query_genres))>=threshold)

potential_movies = movies[matches]

movieId_rating_df = ratings.loc[:,['movieId','rating']]
movieId_avg_rating_df = movieId_rating_df.groupby('movieId',as_index=False).mean()

if query_movie_id not in movieId_avg_rating_df['movieId'].tolist():
    print("Cannot be given any recomendations")

available_potential_movies = pd.merge(potential_movies, movieId_avg_rating_df, how='inner', on=['movieId'])

query_movie_rating = mean(ratings[ratings.movieId==query_movie_id].rating)
rating_similarity = [(1-(abs(query_movie_rating-x))/5)>=0.8 for x in available_potential_movies.rating]

potential_movies2 = available_potential_movies[rating_similarity]

del movieId_avg_rating_df, available_potential_movies, potential_movies, movieId_rating_df

available_potential_movies2 = pd.merge(potential_movies2, movieId_tags_joined_df_cleaned, how='inner', on=['movieId'])

query_tag = movieId_tags_joined_df_cleaned[movieId_tags_joined_df_cleaned.movieId==query_movie_id].iloc[0,1]

tags_list = list(available_potential_movies2.cleaned_tags)
tags_similarity = get_similarity(lsi, dictionary, tags_list, query_tag)
available_potential_movies2['tag_sim'] = tags_similarity

available_potential_movies2.sort_values(by=['tag_sim'], inplace=True, ascending=False)

del available_potential_movies2['genre_id_list'], available_potential_movies2['cleaned_tags']

recommendation = available_potential_movies2.head(11)

recommendation.to_csv(output_folder+str(query_movie_name)+'.csv', index = False)

end_time = "############### " + ptime()+" RESULT IS SAVEDDD ###############"

print(start_time)
print(end_time)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
