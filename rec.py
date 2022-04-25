# import pandas as pd
# import json
# from sklearn.manifold import TSNE
# from sklearn.metrics.pairwise import cosine_similarity

# # Data Preprocessing
# columns = ['user_id', 'item_id', 'rating', 'timestamp']
# df = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)

# columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
#            'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
#            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# movies = pd.read_csv('ml-100k/u.item', sep='|',
#                      names=columns, encoding='latin-1')
# movie_names = movies[['item_id', 'movie title']]

# # Normalize the data
# combined_movies_data = pd.merge(df, movie_names, on='item_id')
# rating_data = combined_movies_data.pivot_table(
#     values='rating', index='user_id', columns='movie title', fill_value=0)
# X = rating_data.T
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(X)

# # Recommending using Cosine Similarity
# # answer = input('Your Title : ')


# def top_ten(answer):
#     sim = cosine_similarity(tsne_results)
#     col_idx = rating_data.columns.get_loc(
#         answer)
#     sim_specific = sim[col_idx]
#     result = pd.DataFrame({'Similarity': sim_specific, 'Top Ten Movies': rating_data.columns}).sort_values(
#         'Similarity', ascending=False).head(10).to_dict()
#     movies = result['Top Ten Movies']
#     return movies


# # print(top_ten(answer))

# # top_ten = pd.DataFrame({'Top Ten Movies': {
# #                        'Similarity': sim_specific, 'Movies': rating_data.columns}}).head(10).to_dict()


# # Dump into JSON
# # top_ten_dump = json.dump(top_ten)

# from tqdm import tqdm
# import numpy as np
# import pandas as pd

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# from surprise import NormalPredictor, SVD, KNNBasic, NMF
# from surprise import Dataset, Reader, accuracy
# from surprise.model_selection import cross_validate, KFold

# """"DATA PREPROCESSING"""
# rating_data = 'goodreads/ratings.csv'
# book_data = 'goodreads/books.csv'

# # Load raw csv into dataframe
# df_ratings = pd.read_csv(rating_data)
# df_books = pd.read_csv(book_data)

# """GET SUBSET DATA"""


# def get_subset(df, number):
#     rids = np.arange(df.shape[0])
#     np.random.shuffle(rids)
#     df_subset = df.iloc[rids[:number], :].copy()
#     return df_subset


# df_ratings_100k = get_subset(df_ratings, 100000)
# df_books_1000 = get_subset(df_books, 1000)

# """READ RATINGS"""
# # Surprise reader
# reader = Reader(rating_scale=(0, 5))

# # Finally load all ratings
# ratings = Dataset.load_from_df(df_ratings_100k, reader)

# # """COLLABORATIVE FILTERING | EVALUATING MODEL"""
# # kf = KFold(n_splits=5)

# # algos = [SVD(), NMF(), KNNBasic()]
# # # SVD : Singular Value Decomposition
# # # NMF : Non-negative Matrix Factorization


# # def get_rmse(algo, testset):
# #     pred = algo.test(testset)
# #     accuracy.rmse(pred, verbose=True)


# # def get_mae(algo, testset):
# #     pred = algo.test(testset)
# #     accuracy.mae(pred, verbose=True)


# # for trainset, testset in tqdm(kf.split(ratings)):
# #     """
# #         Get evaluation with cross-validation for different algorithms.
# #     """
# #     for algo in algos:
# #         algo.fit(trainset)
# #         get_rmse(algo, testset)
# #         get_mae(algo, testset)

# """CONTENT-BASED FILTERING"""
# # computing similarities
# df_books_1000 = df_books_1000.reset_index(drop=True)
# df_books_1000.head()
# df_books_1000 = df_books_1000.dropna()
# df_books_1000 = df_books_1000.reset_index(drop=True)

# # compute a TFIDF on the title of the books
# tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3),
#                      min_df=0, stop_words='english')
# tfidf_matrix = tf.fit_transform(df_books_1000['original_title'])

# # compute linear kernel and cosine similarities
# lin_kernel = linear_kernel(tfidf_matrix, tfidf_matrix)
# cos_similar = cosine_similarity(tfidf_matrix, tfidf_matrix)

# # generate in 'results' the most similar books for each books: put a pair (score, book_id)
# results = {}
# for idx, row in df_books_1000.iterrows():
#     similar_indices = lin_kernel[idx].argsort()[:-100:-1]
#     similar_items = [(lin_kernel[idx][i], df_books_1000['book_id'].iloc[[
#                       i]].tolist()[0]) for i in similar_indices]
#     results[idx] = similar_items[1:]

# # transform a 'book_id' into its corresponding book title

# def item(id):
#     return df_books_1000.loc[df_books_1000['book_id'] == id]['original_title'].tolist()[0].split(' - ')[0]

# # transform a 'book_id' into the index id

# def get_idx(id):
#     return df_books_1000[df_books_1000['book_id'] == id].index.tolist()[0]

# # put everything together here:

# def recommend(answer):
#     num = 10
#     print("Recommending " + str(num) +
#           " products similar to " + item(answer) + "...")
#     print("-------")
#     recs = results[get_idx(answer)][:num]
#     for rec in recs:
#         print("\tRecommended: " +
#               item(rec[1]) + " (score:" + str(rec[0]) + ")")

# import numpy as np
# import pandas as pd

# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer

# data = pd.read_csv('goodbooks-10k/books.csv')

# #Extract relevant columns that would influence a book's rating based on book title.
# books_title = data[['book_id', 'title']]
# books_title.head()

# #initialize vectorizer
# vect = CountVectorizer(analyzer = 'word', ngram_range = (1,2), stop_words = 'english', min_df = 0.002) #min_df = rare words, max_df = most used words
# #ngram_range = (1,2) - if used more than  1(value), lots of features or noise

# #Fit into the title
# vect.fit(books_title['title'])
# title_matrix = vect.transform(books_title['title'])
# #Lets find vocabulary/features
# features = vect.get_feature_names()
# cosine_sim_titles = cosine_similarity(title_matrix, title_matrix)
# #Get books which are similar to a given title
# title_id = 100
# books_title['title'].iloc[title_id]
# #Find out what features have been considered  by the vectorizer for a given title ?
# feature_array = np.squeeze(title_matrix[title_id].toarray()) #squeeze activity matrix into array
# idx = np.where(feature_array > 0)
# idx[0]
# [features[x] for x in idx[0]]
# idx[0]

# #Cosine similarity with other similar titles
# n = 10 #how many books to be recommended
# top_n_idx = np.flip(np.argsort(cosine_sim_titles[title_id,]), axis = 0)[0:n]
# top_n_sim_values = cosine_sim_titles[title_id, top_n_idx]
# top_n_sim_values

# #find top n with values > 0
# top_n_idx = top_n_idx[top_n_sim_values > 0]
# #Matching books
# books_title['title'].iloc[top_n_idx]

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel

# tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1,2), min_df = 0, stop_words = 'english')
# tfidf_matrix = tf.fit_transform(books_title['title'])
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# titles = books_title['title']
# indices = pd.Series(books_title.index, index = books_title['title']) #converting all titles into a Series

# #Function that gets book recommendations based on the cosine similarity score of book titles
# def book_recommendations(title, n):
#     idx = indices[title]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)
#     sim_scores = sim_scores[1:n+1]
#     book_indices = [i[0] for i in sim_scores]
#     return titles.iloc[book_indices]

# #Recommend n books for a book having index 1
# book_index = 9
# n = 10

# print(books_title['title'][book_index])
# book_recommendations(books_title.title[book_index],n)


from surprise.model_selection import cross_validate, KFold
from surprise import Dataset, Reader, accuracy
from surprise import NormalPredictor, SVD, KNNBasic, NMF
import tqdm
import re
import pickle
import operator
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import csr_matrix
from pandas.api.types import is_numeric_dtype
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore")

"""DATASETS LOADING"""

books = pd.read_csv(r"Datasets/Books.csv", delimiter=';',
                    error_bad_lines=False, encoding='ISO-8859-1', warn_bad_lines=False)
users = pd.read_csv(r"Datasets/Users.csv", delimiter=';',
                    error_bad_lines=False, encoding='ISO-8859-1', warn_bad_lines=False)
ratings = pd.read_csv(r"Datasets/Book-Ratings.csv", delimiter=';',
                      error_bad_lines=False, encoding='ISO-8859-1', warn_bad_lines=False)

# print("Books Data:    ", books.shape)
# print("Users Data:    ", users.shape)
# print("Books-ratings: ", ratings.shape)

"""DATA PREPROCESSING"""
# Drop URL columns
books.drop(['Image-URL-S', 'Image-URL-L'], axis=1, inplace=True)
# Checking for null values
books.isnull().sum()
books.loc[books['Book-Author'].isnull(), :]
books.loc[books['Publisher'].isnull(), :]

# Handling the null values
books.at[187689, 'Book-Author'] = 'Other'

books.at[128890, 'Publisher'] = 'Other'
books.at[129037, 'Publisher'] = 'Other'

pd.set_option('display.max_colwidth', -1)
books.at[209538, 'Publisher'] = 'DK Publishing Inc'
books.at[209538, 'Year-Of-Publication'] = 2000
books.at[209538,
         'Book-Title'] = 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)'
books.at[209538, 'Book-Author'] = 'Michael Teitelbaum'

books.at[221678, 'Publisher'] = 'DK Publishing Inc'
books.at[221678, 'Year-Of-Publication'] = 2000
books.at[209538,
         'Book-Title'] = 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'
books.at[209538, 'Book-Author'] = 'James Buckley'

books.at[220731, 'Publisher'] = 'Gallimard'
books.at[220731, 'Year-Of-Publication'] = '2003'
books.at[209538, 'Book-Title'] = 'Peuple du ciel - Suivi de Les bergers '
books.at[209538, 'Book-Author'] = 'Jean-Marie Gustave Le ClÃ?Â©zio'

# Converting year of publication in Numbers
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
# Replacing Invalid years with max year
count = Counter(books['Year-Of-Publication'])
[k for k, v in count.items() if v == max(count.values())]
books.loc[books['Year-Of-Publication'] > 2021, 'Year-Of-Publication'] = 2002
books.loc[books['Year-Of-Publication'] == 0, 'Year-Of-Publication'] = 2002
# Uppercasing all alphabets in ISBN
books['ISBN'] = books['ISBN'].str.upper()
# Drop duplicate rows
books.drop_duplicates(keep='last', inplace=True)
books.reset_index(drop=True, inplace=True)


required = users[users['Age'] <= 80]
required = required[required['Age'] >= 10]
mean = round(required['Age'].mean())
# outliers with age grater than 80 are substituted with mean
users.loc[users['Age'] > 80, 'Age'] = mean
# outliers with age less than 10 years are substitued with mean
users.loc[users['Age'] < 10, 'Age'] = mean
users['Age'] = users['Age'].fillna(mean)  # filling null values with mean
users['Age'] = users['Age'].astype(int)  # changing Datatype to int

list_ = users.Location.str.split(', ')

city = []
state = []
country = []
count_no_state = 0
count_no_country = 0

for i in range(0, len(list_)):
    # removing invalid entries too
    if list_[i][0] == ' ' or list_[i][0] == '' or list_[i][0] == 'n/a' or list_[i][0] == ',':
        city.append('other')
    else:
        city.append(list_[i][0].lower())

    if(len(list_[i]) < 2):
        state.append('other')
        country.append('other')
        count_no_state += 1
        count_no_country += 1
    else:
        # removing invalid entries
        if list_[i][1] == ' ' or list_[i][1] == '' or list_[i][1] == 'n/a' or list_[i][1] == ',':
            state.append('other')
            count_no_state += 1
        else:
            state.append(list_[i][1].lower())

        if(len(list_[i]) < 3):
            country.append('other')
            count_no_country += 1
        else:
            if list_[i][2] == '' or list_[i][1] == ',' or list_[i][2] == ' ' or list_[i][2] == 'n/a':
                country.append('other')
                count_no_country += 1
            else:
                country.append(list_[i][2].lower())

users = users.drop('Location', axis=1)

temp = []
for ent in city:
    # handling cases where city/state entries from city list as state is already given
    c = ent.split('/')
    temp.append(c[0])

df_city = pd.DataFrame(temp, columns=['City'])
df_state = pd.DataFrame(state, columns=['State'])
df_country = pd.DataFrame(country, columns=['Country'])

users = pd.concat([users, df_city], axis=1)
users = pd.concat([users, df_state], axis=1)
users = pd.concat([users, df_country], axis=1)

# Drop duplicate rows
users.drop_duplicates(keep='last', inplace=True)
users.reset_index(drop=True, inplace=True)


# checking ISBN
flag = 0
k = []
reg = "[^A-Za-z0-9]"

for x in ratings['ISBN']:
    z = re.search(reg, x)
    if z:
        flag = 1

# if flag == 1:
#     print("False")
# else:
#     print("True")

# removing extra characters from ISBN (from ratings dataset) existing in books dataset
bookISBN = books['ISBN'].tolist()
reg = "[^A-Za-z0-9]"
for index, row_Value in ratings.iterrows():
    z = re.search(reg, row_Value['ISBN'])
    if z:
        f = re.sub(reg, "", row_Value['ISBN'])
        if f in bookISBN:
            ratings.at[index, 'ISBN'] = f

# Uppercasing all alphabets in ISBN
ratings['ISBN'] = ratings['ISBN'].str.upper()

# Drop duplicate rows
ratings.drop_duplicates(keep='last', inplace=True)
ratings.reset_index(drop=True, inplace=True)


"""MERGING TABLES"""
dataset = pd.merge(books, ratings, on='ISBN', how='inner')
dataset = pd.merge(dataset, users, on='User-ID', how='inner')

# Explicit Ratings Dataset
dataset1 = dataset[dataset['Book-Rating'] != 0]
dataset1 = dataset1.reset_index(drop=True)
dataset1.shape

# Implicit Ratings Dataset
dataset2 = dataset[dataset['Book-Rating'] == 0]
dataset2 = dataset2.reset_index(drop=True)
dataset2.shape

# """EVALUATING MODEL"""


# """GET SUBSET DATA"""


# def get_subset(df, number):
#     rids = np.arange(df.shape[0])
#     np.random.shuffle(rids)
#     df_subset = df.iloc[rids[:number], :].copy()
#     return df_subset


# df_ratings_100k = get_subset(ratings, 100000)

# """READ RATINGS"""
# # Surprise reader
# reader = Reader(rating_scale=(0, 5))

# # Finally load all ratings
# df_ratings = Dataset.load_from_df(df_ratings_100k, reader)

# """COLLABORATIVE FILTERING | EVALUATING MODEL"""
# kf = KFold(n_splits=3)

# algos = [SVD(), NMF(), KNNBasic()]
# # SVD : Singular Value Decomposition
# # NMF : Non-negative Matrix Factorization


# def get_rmse(algo, testset):
#     pred = algo.test(testset)
#     accuracy.rmse(pred, verbose=True)


# def get_mae(algo, testset):
#     pred = algo.test(testset)
#     accuracy.mae(pred, verbose=True)


# for trainset, testset in tqdm(kf.split(df_ratings)):
#     """
#         Get evaluation with cross-validation for different algorithms.
#     """
#     for algo in algos:
#         algo.fit(trainset)
#         get_rmse(algo, testset)
#         get_mae(algo, testset)

"""RECOMMENDER SYSTEM"""
# bookName = "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))"
number = 5

"""1. Popularity Based (Top In whole collection)"""


def popularity_based(dataframe, n):
    if n >= 1 and n <= len(dataframe):
        data = pd.DataFrame(dataframe.groupby('ISBN')[
                            'Book-Rating'].count()).sort_values('Book-Rating', ascending=False).head(n)
        result = pd.merge(data, books, on='ISBN')
        return result
    return "Invalid number of books entered!!"


# print("Top", number, "Popular books are: ")
pop_based_colle = popularity_based(dataset1, number)

"""2. Popularity Based (Top In a given place)"""


# def search_unique_places(dataframe, place):
#     place = place.lower()

#     if place in list(dataframe['City'].unique()):
#         return dataframe[dataframe['City'] == place]
#     elif place in list(dataframe['State'].unique()):
#         return dataframe[dataframe['State'] == place]
#     elif place in list(dataframe['Country'].unique()):
#         return dataframe[dataframe['Country'] == place]
#     else:
#         return "Invalid Entry"


# place = input("Enter the name of place: ")
# data = search_unique_places(dataset1, place)

# if isinstance(data, pd.DataFrame):
#     data = popularity_based(data, number)

"""3. Books by same author, publisher of given book name"""


# def printBook(k, n):
#     z = k['Book-Title'].unique()
#     for x in range(len(z)):
#         print(z[x])
#         if x >= n-1:
#             break


# def get_books(dataframe, name, n):
#     print("\nBooks by same Author:\n")
#     au = dataframe['Book-Author'].unique()

#     data = dataset1[dataset1['Book-Title'] != name]

#     if au[0] in list(data['Book-Author'].unique()):
#         k2 = data[data['Book-Author'] == au[0]]
#     k2 = k2.sort_values(by=['Book-Rating'])
#     printBook(k2, n)

#     print("\n\nBooks by same Publisher:\n")
#     au = dataframe['Publisher'].unique()

#     if au[0] in list(data['Publisher'].unique()):
#         k2 = pd.DataFrame(data[data['Publisher'] == au[0]])
#     k2 = k2.sort_values(by=['Book-Rating'])
#     printBook(k2, n)


# if bookName in list(dataset1['Book-Title'].unique()):
#     d = dataset1[dataset1['Book-Title'] == bookName]
#     get_books(d, bookName, number)
# else:
#     print("Invalid Book Name!!")

"""4. Books popular Yearly"""

# data = pd.DataFrame(dataset1.groupby(
#     'ISBN')['Book-Rating'].count()).sort_values('Book-Rating', ascending=False)
# data = pd.merge(data, books, on='ISBN')

# years = set()
# indices = []
# for ind, row in data.iterrows():
#     if row['Year-Of-Publication'] in years:
#         indices.append(ind)
#     else:
#         years.add(row['Year-Of-Publication'])

# data = data.drop(indices)
# data = data.drop('Book-Rating', axis=1)
# data = data.sort_values('Year-Of-Publication')

# pd.set_option("display.max_rows", None, "display.max_columns", None)

"""5. AVERAGE WEIGHTED RATINGS """


# def avgRating(newdf, df):
#     newdf['Average Rating'] = 0
#     for x in range(len(newdf)):
#         l = list(df.loc[df['Book-Title'] ==
#                  newdf['Book-Title'][x]]['Book-Rating'])
#         newdf['Average Rating'][x] = sum(l)/len(l)
#     return newdf


# df = pd.DataFrame(dataset1['Book-Title'].value_counts())
# df['Total-Ratings'] = df['Book-Title']
# df['Book-Title'] = df.index
# df.reset_index(level=0, inplace=True)
# df = df.drop('index', axis=1)

# # df = avgRating(df, dataset1)
# # df.to_pickle('weightedData')
# df = pd.read_pickle('weightedData')
# # C - Mean vote across the whole
# C = df['Average Rating'].mean()

# # Minimum number of votes required to be in the chart
# m = df['Total-Ratings'].quantile(0.90)


# def weighted_rating(x, m=m, C=C):
#     v = x['Total-Ratings']  # v - number of votes
#     R = x['Average Rating']  # R - Average Rating
#     return (v/(v+m) * R) + (m/(m+v) * C)


# df = df.loc[df['Total-Ratings'] >= m]

# df['score'] = df.apply(weighted_rating, axis=1)
# df = df.sort_values('score', ascending=False)

# print("Recommended Books:-\n")
# df.head(number)

"""6. Collaborative Filtering (User-Item Filtering)"""

df = pd.DataFrame(dataset1['Book-Title'].value_counts())
df['Total-Ratings'] = df['Book-Title']
df['Book-Title'] = df.index
df.reset_index(level=0, inplace=True)
df = df.drop('index', axis=1)

df = dataset1.merge(df, left_on='Book-Title',
                    right_on='Book-Title', how='left')
df = df.drop(['Year-Of-Publication', 'Publisher',
             'Age', 'City', 'State', 'Country'], axis=1)

popularity_threshold = 50
popular_book = df[df['Total-Ratings'] >= popularity_threshold]
popular_book = popular_book.reset_index(drop=True)

testdf = pd.DataFrame()
testdf['ISBN'] = popular_book['ISBN']
testdf['Book-Rating'] = popular_book['Book-Rating']
testdf['User-ID'] = popular_book['User-ID']
testdf = testdf[['User-ID', 'Book-Rating']].groupby(testdf['ISBN'])

listOfDictonaries = []
indexMap = {}
reverseIndexMap = {}
ptr = 0

for groupKey in testdf.groups.keys():
    tempDict = {}
    groupDF = testdf.get_group(groupKey)
    for i in range(0, len(groupDF)):
        tempDict[groupDF.iloc[i, 0]] = groupDF.iloc[i, 1]
    indexMap[ptr] = groupKey
    reverseIndexMap[groupKey] = ptr
    ptr = ptr+1
    listOfDictonaries.append(tempDict)

dictVectorizer = DictVectorizer(sparse=True)
vector = dictVectorizer.fit_transform(listOfDictonaries)
pairwiseSimilarity = cosine_similarity(vector)


def printBookDetails(bookID):
    print(dataset1[dataset1['ISBN'] == bookID]['Book-Title'].values[0])
    """
    print("Title:", dataset1[dataset1['ISBN']==bookID]['Book-Title'].values[0])
    print("Author:",dataset1[dataset['ISBN']==bookID]['Book-Author'].values[0])
    #print("Printing Book-ID:",bookID)
    print("\n")
    """


def getTopRecommandations(bookID):
    collaborative = []
    row = reverseIndexMap[bookID]
    print("Input Book:")
    printBookDetails(bookID)

    print("\nRECOMMENDATIONS:\n")

    mn = 0
    similar = []
    for i in np.argsort(pairwiseSimilarity[row])[:-2][::-1]:
        if dataset1[dataset1['ISBN'] == indexMap[i]]['Book-Title'].values[0] not in similar:
            if mn >= number:
                break
            mn += 1
            similar.append(dataset1[dataset1['ISBN'] ==
                           indexMap[i]]['Book-Title'].values[0])
            printBookDetails(indexMap[i])
            collaborative.append(
                dataset1[dataset1['ISBN'] == indexMap[i]]['Book-Title'].values[0])
    return collaborative


k = list(dataset1['Book-Title'])
m = list(dataset1['ISBN'])


"""7. Correlation Based"""

# popularity_threshold = 50

# user_count = dataset1['User-ID'].value_counts()
# data = dataset1[dataset1['User-ID'].isin(
#     user_count[user_count >= popularity_threshold].index)]
# rat_count = data['Book-Rating'].value_counts()
# data = data[data['Book-Rating'].isin(rat_count[rat_count >=
#                                      popularity_threshold].index)]

# matrix = data.pivot_table(index='User-ID', columns='ISBN',
#                           values='Book-Rating').fillna(0)

# average_rating = pd.DataFrame(dataset1.groupby('ISBN')['Book-Rating'].mean())
# average_rating['ratingCount'] = pd.DataFrame(
#     ratings.groupby('ISBN')['Book-Rating'].count())
# average_rating.sort_values('ratingCount', ascending=False).head()

# isbn = books.loc[books['Book-Title'] ==
#                  bookName].reset_index(drop=True).iloc[0]['ISBN']
# row = matrix[isbn]
# correlation = pd.DataFrame(matrix.corrwith(row), columns=['Pearson Corr'])
# corr = correlation.join(average_rating['ratingCount'])

# res = corr.sort_values(
#     'Pearson Corr', ascending=False).head(number+1)[1:].index
# corr_books = pd.merge(pd.DataFrame(res, columns=['ISBN']), books, on='ISBN')

"""8. Nearest Neighbours Based"""

# data = (dataset1.groupby(by=['Book-Title'])['Book-Rating'].count().reset_index().
#         rename(columns={'Book-Rating': 'Total-Rating'})[['Book-Title', 'Total-Rating']])

# result = pd.merge(data, dataset1, on='Book-Title')
# result = result[result['Total-Rating'] >= popularity_threshold]
# result = result.reset_index(drop=True)

# matrix = result.pivot_table(
#     index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)
# up_matrix = csr_matrix(matrix)

# model = NearestNeighbors(metric='cosine', algorithm='brute')
# model.fit(up_matrix)

# distances, indices = model.kneighbors(
#     matrix.loc[bookName].values.reshape(1, -1), n_neighbors=number+1)
# print("\nRecommended books:\n")
# for i in range(0, len(distances.flatten())):
#     if i > 0:
#         print(matrix.index[indices.flatten()[i]])


"""9. Content Based"""


def content_based(answer):
    popularity_threshold = 80
    popular_book = df[df['Total-Ratings'] >= popularity_threshold]
    popular_book = popular_book.reset_index(drop=True)
    popular_book.shape

    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english')
    tfidf_matrix = tf.fit_transform(popular_book['Book-Title'])
    tfidf_matrix.shape

    normalized_df = tfidf_matrix.astype(np.float32)
    cosine_similarities = cosine_similarity(normalized_df, normalized_df)
    cosine_similarities.shape

    isbn = books.loc[books['Book-Title'] ==
                     answer].reset_index(drop=True).iloc[0]['ISBN']
    content = []

    idx = popular_book.index[popular_book['ISBN'] == isbn].tolist()[0]
    similar_indices = cosine_similarities[idx].argsort()[::-1]
    similar_items = []
    for i in similar_indices:
        if popular_book['Book-Title'][i] != answer and popular_book['Book-Title'][i] not in similar_items and len(similar_items) < number:
            similar_items.append(popular_book['Book-Title'][i])
            content.append(popular_book['Book-Title'][i])


"""10. Hybrid Approach (Content+Collaborative) Using percentile"""

# z = list()
# k = float(1/number)
# for x in range(number):
#     z.append(1-k*x)

# dictISBN = {}
# for x in collaborative:
#     dictISBN[x] = z[collaborative.index(x)]

# for x in content:
#     if x not in dictISBN:
#         dictISBN[x] = z[content.index(x)]
#     else:
#         dictISBN[x] += z[content.index(x)]

# ISBN = dict(sorted(dictISBN.items(), key=operator.itemgetter(1), reverse=True))
# w = 0
# print("Input Book:\n")
# print(bookName)
# print("\nRecommended Books:\n")
# for x in ISBN.keys():
#     if w >= number:
#         break
#     w += 1
#     print(x)
