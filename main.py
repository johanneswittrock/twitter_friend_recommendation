import tweepy as tw
import re
from textblob import TextBlob
import numpy
import matrixfactorization as mf
import json
from sklearn.metrics.pairwise import cosine_similarity
import sys
import pandas as pd

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
        sentiment_value = (((sentiment_value_temp - (-1)) * 4) / 2) + 1

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
        sentiment_value = (((sentiment_value_temp - (-1)) * 4) / 2) + 1


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
        sentiment_value = (((sentiment_value_temp - (-1)) * 4) / 2) + 1

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
        sentiment_value = (((sentiment_value_temp - (-1)) * 4) / 2) + 1

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
        sentiment_value = (((sentiment_value_temp - (-1)) * 4) / 2) + 1

        if user in users:
            i = users[user]
            matrix[i][4] = sentiment_value
        else:
            users[user] = counter
            counter += 1
            matrix.append([0,0,0,0,sentiment_value])


def mft(R):
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
    return nR



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
              since=date_since).items(20)

movieTweets = tw.Cursor(api.search,
              q=movies,
              lang="en",
              since=date_since).items(20)
    
footballTweets = tw.Cursor(api.search,
              q=football,
              lang="en",
              since=date_since).items(20)

trumpTweets = tw.Cursor(api.search,
              q=trump,
              lang="en",
              since=date_since).items(20)

carTweets = tw.Cursor(api.search,
              q=cars,
              lang="en",
              since=date_since).items(20)

preprocessClimateTweets(climateTweets)
preprocessMovieTweets(movieTweets)
preprocessFootballTweets(footballTweets)
preprocessTrumpTweets(trumpTweets)
preprocessCarTweets(carTweets)

np_array = numpy.array(matrix)

#test = [[5,3,0,2,1],
#[2,0,3,0,5],
#[1,4,5,2,0],
#[0,5,3,2,5],
#[5,3,0,0,1]]
#np_array= numpy.array(test)

f = open("oldmatrix.txt", "w")
f.write(numpy.array2string(np_array))
f.close

rating_matrix = mft(np_array)
f = open("matrix.txt", "w")
f.write(numpy.array2string(rating_matrix))
f.close

df = pd.DataFrame(rating_matrix)
cos_sim = cosine_similarity(df,df)
f = open("cosinesimilarity.txt", "w")
f.write(numpy.array2string(cos_sim))
f.close
