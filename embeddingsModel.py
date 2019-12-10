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

def get_weight_matrix2(embedding, vocab):
    vocab_size2 = len(vocab) + 1
    weight_matrix = zeros((vocab_size2, 200))
    for word, i in vocab:
        vector = None
        try:
            vector = embedding.get_vector(word)
        except:
            continue
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix
def createModelC(max_length, embedding_layer):
    filter_sizes = (1,2,3,4)
    input_shape = (max_length,)
    num_filters = 100

    model_input = Input(shape=input_shape)
    zz = embedding_layer(model_input)

    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(zz)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks if len(conv_blocks) > 1 else conv_blocks[0])
    z = Dropout(0.5)(z)
    model_output = Dense(10, activation="sigmoid", bias_initializer='zeros')(z)
    model_output = Dense(10)(model_output)
    model_output = Dropout(0.5)(model_output)
    model_output = Dense(3, activation="selu")(model_output)
    model = Model(model_input, model_output)
    max = 0
    return model
def convModel(tweets, stances, tweets_test, stances_test):
    #General Parameters
    global max
    embeding_dim = 200
    dropout_prob = (0.0, 0.5)
    batch_size = 64
    num_epochs = 20

    print('Fitting tokenizer')
    tokenizer = Tokenizer()
    tokenizer.fit_on_sequences(tweets + tweets2)
    max_length = max([len(s.split()) for s in tweets + tweets2])
    print('max_length', max_length)

    vocab_size = len(tokenizer.word_index) + 1

    #Train and test split
    print('Train and test split')
    x_train, x_test, y_train, y_test = train_test_split(tweets, stances, test_size=0.2)
    print('x_train: ', len(x_train), 'x_test', len(x_test))


    #Training data
    #traindata = np.array(x_train)
    #testdata = np.array(x_test)

    trainTokens = tokenizer.texts_to_sequences(x_train)
    Xtrain = pad_sequences(trainTokens, maxlen=max_length, padding='post')
    XtestTokens = tokenizer.texts_to_sequences(x_test)
    Xtest = pad_sequences(XtestTokens, maxlen=max_length, padding='post')
    #============ TEST DATA =============================================
    #testgroup = np.array(tweets_test)
    #testGroupTokens = tokenizer.texts_to_sequences(tweets_test)
    #XtestGroup = pad_sequences(testGroupTokens, maxlen=max_length, padding='post')
    #print('Xtrain padding: ', len(Xtrain), 'Xtest padding: ', len(Xtest), 'XtestGroup padding: ', len(XtestGroup))

    #Convert stances to categorical output
    y_test = np_utils.to_categorical(y_test, num_classes=3)
    y_train = np_utils.to_categorical(y_train, num_classes=3)
    y_testGroup = np_utils.to_categorical(stances_test, num_classes=3)
    print('y_test: ', len(y_test), 'y_train: ', len(y_train), 'y_testGroup: ', len(stances_test))


    print('Loading embeddings..')
    #load word2vec and create embedding layer
    wv_from_bin = KeyedVectors.load_word2vec_format(datapath('E:/glove/glove.twitter.27B.200dGINSIM.txt'),binary=False)
    embedding_vectors = get_weight_matrix2(wv_from_bin, tokenizer.word_index.items())
    embedding_layer = Embedding(vocab_size, embeding_dim, weights=[embedding_vectors], input_length=max_length, trainable=False)

    #Create the model
    print('Create and compile the model..')
    model = createModelC(max_length, embedding_layer)
    model.compile(loss="categorical_hinge", optimizer="adam", metrics=[f1])
    model.summary(85)

    print('Fitting the model..')
    history = model.fit(Xtrain, y_train, batch_size=batch_size, epochs=num_epochs,
                        validation_data=(Xtest, y_test), verbose=2)
    print('History', history.history)

    # evaluate
    print('Predicting (training)..')
    ypred = model.predict(Xtest)
    print('Accuracy (TRAIN): %f' % (model.evaluate(Xtest,y_test)[0]*100))
    print('FScore (TRAIN): %f' % (f1(y_test, ypred)*100))

    print('Predicting (testing)..')
    #ypred = model.predict(XtestGroup)
    #print('Accuracy (TEST): %f' % (model.evaluate(XtestGroup,y_testGroup)[0]*100))
    #print('FScore (TEST): %f' % (f1(y_testGroup,ypred)*100))


indata = readfile('SemEval2016-Task6-subtaskA-traindata-gold.csv', False)
data = preprocesstweets(indata,ignoreNONE=False, version =0)
tweets = [' '.join(d[0]) for d in data]
stances = [d[7] for d in data]
encoder = LabelEncoder()
stances = encoder.fit_transform(stances)

indata = readfile('SemEval2016-Task6-subtaskA-testdata-gold.txt', True)
data = preprocesstweets(indata,ignoreNONE=False, version =1)
tweets2 = [' '.join(d[0]) for d in data]
stances2 = [d[7] for d in data]
stances2 = encoder.fit_transform(stances2)


convModel(tweets, stances, tweets2, stances2)
#The model is not working currently, last edits caused a problem. Reported results were from previous stage