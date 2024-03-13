import pandas
import string
import nltk
import keras
import sklearn
import time
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import numpy as np
from time import time
from nltk.corpus import stopwords
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
# from keras.preprocessing import 


# load dataset
dataframe = pandas.read_csv("tet.csv", header=0)
dataframe.drop_duplicates(inplace = True)
data= dataframe
print(dataframe.head())

print(type(data['text']))

train_size = int(len(data) * .8)

print(int(len(data['text'])))
print(train_size)

texts= data['text']
tags = data['class']

train_posts = data['text'][:train_size]
train_tags = data['class'][:train_size]

test_posts = data['text'][train_size:]
test_tags =  data['class'][train_size:]

tokenizer = Tokenizer(num_words=None,lower=False)
tokenizer.fit_on_texts(texts)

x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')

encoder = LabelEncoder()
encoder.fit(tags)
tagst=encoder.fit_transform(tags)

num_classes = int((len(set(tagst))))
print((len(set(tagst))))

y_train = encoder.fit_transform(train_tags)
y_test = encoder.fit_transform(test_tags)

y_train= keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


num_labels = int(len(y_train.shape))
vocab_size = len(tokenizer.word_index) + 1

max_words=vocab_size


import keras.backend as K
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

from keras.metrics import Precision , Recall , Accuracy , TruePositives , TrueNegatives , FalsePositives , FalseNegatives

# Build the model
model = Sequential()
model.add(Dense(1024, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy','Recall','Precision', f1_metric,'TruePositives','TrueNegatives','FalsePositives','FalseNegatives'])


batch_size = 100
epochs = 2

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


model.save('my_model.h1')

import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#model = keras.models.load_model('my_model.h1')
Evaluation_valus = model.evaluate(x_test,y_test,verbose=0)
print("Loss" , 'categorical_accuracy','Recall','Precision','f1_metric','TruePositives','TrueNegatives','FalsePositives','FalseNegatives')

print(Evaluation_valus)


for x in data["text"][:25]:

    tokens = tokenizer.texts_to_matrix([x], mode='tfidf')

    c=model.predict(np.array(tokens))
    cc=model.predict_classes(tokens)
    xc = encoder.inverse_transform(cc)


    print(c,"= \t",cc,"\t",xc)