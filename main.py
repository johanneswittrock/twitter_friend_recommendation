import os
from numpy.core.defchararray import array
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import tweepy as tw
from nltk.corpus import stopwords
import re
from textblob import TextBlob
import numpy
import sys
import matrixfactorization as mf
import json
import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

numpy.set_printoptions(threshold=sys.maxsize)

def remove_url(txt):
    """Replace URLs found in a text string with nothing 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

j_values = {
    "climate change": 0,
    "movies": 1,
    "football": 2,
    "trump": 3,
    "cars": 4
}

matrix = []
users = {} 
counter = 0



def preprocessClimateTweets(tweets):
    global matrix
    global users
    global counter
    for tweet in tweets:
        user = tweet.user.screen_name
        text = remove_url(tweet.text)
        sentiment_object = TextBlob(text)
        sentiment_value_temp = sentiment_object.sentiment.polarity
        sentiment_value = (((sentiment_value_temp - (-1)) * 5) / 2) 

        if user in users:
            i = users[user]
            matrix[i][0] = sentiment_value
        else:
            users[user] = counter
            counter += 1
            matrix.append([sentiment_value,0,0,0,0])

def preprocessMovieTweets(tweets):
    global matrix
    global users
    global counter
    for tweet in tweets:
        user = tweet.user.screen_name
        text = remove_url(tweet.text)
        sentiment_object = TextBlob(text)
        sentiment_value_temp = sentiment_object.sentiment.polarity
        sentiment_value = (((sentiment_value_temp - (-1)) * 5) / 2) 


        if user in users:
            i = users[user]
            matrix[i][1] = sentiment_value
        else:
            users[user] = counter
            counter += 1
            matrix.append([0,sentiment_value,0,0,0])

def preprocessFootballTweets(tweets):
    global matrix
    global users
    global counter
    for tweet in tweets:
        user = tweet.user.screen_name
        text = remove_url(tweet.text)
        sentiment_object = TextBlob(text)
        sentiment_value_temp = sentiment_object.sentiment.polarity
        sentiment_value = (((sentiment_value_temp - (-1)) * 5) / 2)

        if user in users:
            i = users[user]
            matrix[i][2] = sentiment_value
        else:
            users[user] = counter
            counter += 1
            matrix.append([0,0,sentiment_value,0,0])

def preprocessTrumpTweets(tweets):
    global matrix
    global users
    global counter
    for tweet in tweets:
        user = tweet.user.screen_name
        text = remove_url(tweet.text)
        sentiment_object = TextBlob(text)
        sentiment_value_temp = sentiment_object.sentiment.polarity
        sentiment_value = (((sentiment_value_temp - (-1)) * 5) / 2)

        if user in users:
            i = users[user]
            matrix[i][3] = sentiment_value
        else:
            users[user] = counter
            counter += 1
            matrix.append([0,0,0,sentiment_value,0])

def preprocessCarTweets(tweets):
    global matrix
    global users
    global counter
    for tweet in tweets:
        user = tweet.user.screen_name
        text = remove_url(tweet.text)
        sentiment_object = TextBlob(text)
        sentiment_value_temp = sentiment_object.sentiment.polarity
        #change range of the sentimant value from -1-1 to 0-5
        sentiment_value = (((sentiment_value_temp - (-1)) * 5) / 2)

        if user in users:
            i = users[user]
            matrix[i][4] = sentiment_value
        else:
            users[user] = counter
            counter += 1
            matrix.append([0,0,0,0,sentiment_value])


def mft():
    global np_array
    R = np_array
    # N: num of User
    N = len(R)
    # M: num of Movie
    M = len(R[0])
    #         Num of Features
    K = 3

 
    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

 

    nP, nQ = mf.matrix_factorization(R, P, Q, K)

    nR = numpy.dot(nP, nQ.T)
    f = open("matrix.txt", "w")
    f.write(numpy.array2string(nR))
    f.close


f = open("keys.json")
keys = json.load(f)
consumer_key= keys["consumer_key"]
consumer_secret= keys["consumer_secret"]
access_token= keys["access_token"]
access_token_secret= keys["access_token_secret"]
f.close()

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

#retweets werden rausgefiltert
climatechange = "#climate+change -filter:retweets"
movies = "#movies -filter:retweets"
football = "#football -filter:retweets"
trump = "#trump -filter:retweets"
cars = "#cars -filter:retweets"
date_since = "2021-05-01"

climateTweets = tw.Cursor(api.search,
              q=climatechange,
              lang="en",
              since=date_since).items(3)

movieTweets = tw.Cursor(api.search,
              q=movies,
              lang="en",
              since=date_since).items(3)
    
footballTweets = tw.Cursor(api.search,
              q=football,
              lang="en",
              since=date_since).items(3)

trumpTweets = tw.Cursor(api.search,
              q=trump,
              lang="en",
              since=date_since).items(3)

carTweets = tw.Cursor(api.search,
              q=cars,
              lang="en",
              since=date_since).items(3)

preprocessClimateTweets(climateTweets)
preprocessMovieTweets(movieTweets)
preprocessFootballTweets(footballTweets)
preprocessTrumpTweets(trumpTweets)
preprocessCarTweets(carTweets)

np_array = numpy.array(matrix)
f = open("oldmatrix.txt", "w")
f.write(numpy.array2string(np_array))
f.close
mft()




#tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]

#sentiment_objects = [TextBlob(tweet) for tweet in tweets_no_urls]

#sentiment_objects[0].polarity, sentiment_objects[0]

# Create list of polarity valuesx and tweet text
#sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]

#sentiment_values[0]

# Create dataframe containing the polarity value and tweet text
#sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])

#sentiment_df.head()




#fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
#sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
#            ax=ax,
#             color="purple")

#plt.title("Sentiments from Tweets on Climate Change")
#plt.show()


#am besten wäre es iwie in ein format zu bekommen wie:
#user - tweet - sentiment
#mby nur user und sentiment weil text eig egal ist wenn ich sentiment hab
#topic ist auch wichtig
