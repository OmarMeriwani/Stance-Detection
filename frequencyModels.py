import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from preprocessing import preprocesstweets, readfile, getTfidfRepresentation
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Input, Convolution1D, Embedding, GlobalMaxPooling1D
from keras.models import Sequential
from keras.layers import Flatten
from keras.utils import np_utils
import tensorflow as tf
from evaluation import f1
import pickle
import csv

def createModel(dim):
    model = Sequential()
    model.add(Dense(1400, activation='relu', input_dim=dim)) #64396
    model.add(Dropout(0.3))
    model.add(Dense(600))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.compat.v1.keras.losses.categorical_crossentropy,
                  metrics=[f1])
    return model


def runTest(file1,  version, model, mode='words'):
    """Gets training or test file for stance detection SemiVal 2016 competition and prints prediction results.

        Parameters
        ----------
        file1 : list
            a list with text tokens on index (0)  and hashtags list on index (1)

        istest : Boolean
            specifies if the dataset is for test or training

        version : int
            0: Training dataset, 1: Test dataset, 2:Other domain dataset
        mode : str
            choose either (words) or (hashtags)

        """
    indata = readfile(file1, version)
    data = preprocesstweets(indata,ignoreNONE=False, version =version,lowerCase=True)
    tfidfAdded = getTfidfRepresentation(data, version, mode)
    labels = [d[7] for d in data]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    print(encoder.classes_)
    if version==0:
        x_train, x_test, y_train, y_test = train_test_split(tfidfAdded, y, test_size=0.2)
        y_test = np_utils.to_categorical(y_test, num_classes=3)
        y_train = np_utils.to_categorical(y_train, num_classes=3)
        print(x_train.shape[1])
        print(model.summary())
        model.fit(x_train, y_train, epochs=10, verbose=2, validation_data=(x_test, y_test))
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        ypred = model.predict(x_test)
        print('Training Accuracy: %f' % (acc * 100))
        print('Training F-Score: ', f1(y_test, ypred)*100)
    if version==1 or version==2:
        y = np_utils.to_categorical(y, num_classes=3)
        loss, acc = model.evaluate(tfidfAdded, y)
        ypred = model.predict(tfidfAdded)
        otherdomain = ''
        if version == 2:
            otherdomain = '(other domain)'
        print('TEST Accuracy '+otherdomain+': %f' % ((acc * 100)))
        print('TEST F-Score '+ otherdomain +': ', (f1(y, ypred)*100))

indata = readfile('SemEval2016-Task6-subtaskA-traindata-gold.csv', 0)
data = preprocesstweets(indata,ignoreNONE=False, version =0,lowerCase=True)
tfidfAdded = getTfidfRepresentation(data, 0, 'words')
model = createModel(tfidfAdded.shape[1])

runTest('SemEval2016-Task6-subtaskA-traindata-gold.csv',0,model, 'words')
runTest('SemEval2016-Task6-subtaskA-testdata-gold.txt', 1,model, 'words')
runTest('stance.csv', 2,model, 'words')

