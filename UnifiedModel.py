import numpy as np
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Input, Convolution1D, Embedding, GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from keras.utils import np_utils
from preprocessing import preprocesstweets, readfile, getTfidfRepresentation
from sklearn.preprocessing import LabelEncoder
from evaluation import f1
import pandas as pd
from keras.layers import Flatten
import tensorflow as tf
from evaluation import f1
import pickle
import csv
from frequencyModels import createModel
from embeddingsModel import convModel

modelFrequency = createModel()

