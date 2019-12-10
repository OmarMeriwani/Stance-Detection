import pandas as pd
import numpy as np
import csv
import os
import re
import spacy
from spacy.tokenizer import Tokenizer
from nltk.tokenize import TweetTokenizer
import string
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

nlp = spacy.load("en_core_web_sm")
#tokenizer = Tokenizer(nlp.vocab)
tokenizer = TweetTokenizer()
ps = PorterStemmer()
ls = LancasterStemmer()

def readfile(filename, version=0):
    data = []
    with open(filename, 'r', encoding="iso-8859-1") as fin:
        if version == 1:
            reader = csv.reader(fin, delimiter='\t')
        else:
            reader = csv.reader(fin, quotechar='"')
        columns = next(reader)
        for line in reader:
            data.append(line)
    data_df = pd.DataFrame(data, columns=columns)
    #print( data_df)
    return data_df

def SplitCapitalCase(word):
    if len(word) == 0:
        return
    newWord = word[0]
    for i in range(1, len(word) - 1):
        if (str(word[i]).isupper() and str(word[i + 1]).isupper() == False) or str(word[i]).isnumeric():
            newWord += ' ' + word[i]
        else:
            newWord += word[i]
    newWord += word[len(word) - 1]
    return newWord
def findWordsWithoutSpaces(OriginalWord):
    if OriginalWord[0] == '@' or OriginalWord[0] == '#':
        OriginalWord = OriginalWord[1:]
    else:
        return OriginalWord
    listOfWords = []
    #words with underscore symbol
    underscoresplit = OriginalWord.split('_')
    #words with mixedcase
    for word in underscoresplit:
        if word.strip != '':
            newword = SplitCapitalCase(word)
            if newword is not None:
                for n in newword.split(' '):
                    listOfWords.append(n)
        else:
            continue
    return listOfWords
def preprocesstweets(indata, ignoreNONE, ignorePunctuation=False, ignoreNumbers=False,
                     lowerCase=False, hashTagsMode='dismantled', removeStopWords = False,
                     version = 0):
    data = []
    for i in range(0, len(indata)):
        if version == 0:
            tweet = str(indata.loc[i].values[0])
            stance = str(indata.loc[i].values[2])
            sentiment = str(indata.loc[i].values[3])
            target = str(indata.loc[i].values[1])

        if version == 1:
            tweet = str(indata.loc[i].values[2])
            stance = str(indata.loc[i].values[3])
            sentiment = None
            target = str(indata.loc[i].values[1])

        if version == 2:
            tweet = str(indata.loc[i].values[1])
            target = str(indata.loc[i].values[3])
            sentiment = ''
            stance =str(indata.loc[i].values[4])
            if stance is None or stance == '':
                continue
        hashtags = re.findall(r'\#[a-zA-Z0-9]+\b', tweet)
        mentions = re.findall(r'\@[a-zA-Z0-9]+\b', tweet)

        #if ignore None states is required
        if ignoreNONE and version != 2:
            if stance == 'NONE':
                continue
        #Fixing targets for checking
        targetIsFullyMentioned = False
        targetIsPartiallyMentioned = False
        targetTokens = tokenizer.tokenize(target)
        targetTokens = [w for w in targetTokens if not w in stop_words]
        targetTokensPStems = [ps.stem(w) for w in targetTokens]
        targetTokensLStems = [ls.stem(w) for w in targetTokens]

        #Remove non-ASCII characters
        tweet = tweet.encode('ascii', errors='ignore').decode("utf-8")

        #Ignore numbers
        if ignoreNumbers:
            tweet = [t for t in tweet if t.isnumeric() == False]
        tokens = tokenizer.tokenize(tweet)

        tokensAfterHashtags = []
        #Version with hashtags and mentions
        if hashTagsMode == 'with' or hashTagsMode is None:
            tokensAfterHashtags = tokens

        #Version with dismantled hashtags and mentions
        if hashTagsMode == 'dismantled':
            for token in tokens:
                if token[0] == '#' or token[0] == '@':
                    dismantled = findWordsWithoutSpaces(token)
                    for d in dismantled:
                        tokensAfterHashtags.append(d)
                else:
                    tokensAfterHashtags.append(token)
        #Version without hashtags and mentions
        if hashTagsMode == 'without':
            tokensAfterHashtags = [t for t in tokens if t[0] != '#' and t[0] != '@']
        #Remove punctuation
        if ignorePunctuation:
            tokensAfterHashtags = [token.translate(None, string.punctuation) for token in tokensAfterHashtags if (token[0] != '#' and token[0] != '@') ]
        #LowerCasing
        if lowerCase:
            tokensAfterHashtags = [str(token).lower() for token in tokensAfterHashtags ]
            #tokensAfterHashtags = [token[0]+ str(token[1:]).lower() for token in tokensAfterHashtags if (token[0] == '#' and token[0] == '@') ]
        if removeStopWords:
            tokensAfterHashtags = [w for w in tokensAfterHashtags if not w in stop_words]

        #Find if the target is mentioned
        if ' '.join(tokensAfterHashtags).lower().find(target.lower()) != -1:
            targetIsFullyMentioned = True
        else:
            fullWordExist = len([w for w in tokensAfterHashtags if w in [t for t in targetTokens]])
            #pstemExist = len([w for w in tokensAfterHashtags if w in [ps.stem(t) for t in targetTokensPStems]])
            #lstemExist = len([w for w in tokensAfterHashtags if w in [ls.stem(t) for t in targetTokensLStems]])
            if fullWordExist > 0: #or pstemExist > 0 or lstemExist > 0:
                targetIsPartiallyMentioned = True

        data.append([tokensAfterHashtags,
                     hashtags, mentions, target, targetIsFullyMentioned, targetIsPartiallyMentioned,
                     sentiment, stance])
    return data
def getTfidfRepresentation(data, version=0, mode='words'):
    """Gets list of tokens and hashtags and returns tfidf representation of the textual data.

    Parameters
    ----------
    data : list
        a list with text tokens on index (0)  and hashtags list on index (1)

    istest : Boolean
        specifies if the dataset is for test or training

    mode : str
        choose either (words) or (hashtags) in order to get the tfidf of it

    Returns
    -------
    X : sparse matrix, [n_samples, n_features]
        Tf-idf-weighted document-term matrix.
    """
    textData = [' '.join(row[0]) for row in data]
    hashtagsData = [' '.join(row[1]) for row in data]

    testdata = []
    ngramrange = (1,1)
    if mode == 'words':
        testdata = textData
        ngramrange = (1,3)
    if mode == 'hashtags':
        testdata = hashtagsData
        ngramrange = (1,1)
    if version == 0:
        tf = TfidfVectorizer(analyzer='word',  ngram_range=ngramrange, stop_words="english", lowercase=False,
                             max_features=500000)
        tf_transformer = tf.fit(testdata)
        pickle.dump(tf_transformer, open("tfidf1.pkl", "wb"))

    tf1 = pickle.load(open("tfidf1.pkl", 'rb'))
    tf1_new = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words="english", lowercase=False,
                              max_features=500000, vocabulary=tf1.vocabulary_)
    tfidfdata = tf1_new.fit_transform(testdata)
    return tfidfdata




