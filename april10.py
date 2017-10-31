from __future__ import print_function
import gensim
from gensim.corpora.dictionary import Dictionary

import os
import codecs

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import  re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD,RMSprop
from keras.utils import np_utils

#import numpy as np
import matplotlib as plt
import matplotlib
import os
import theano
from PIL import Image

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from scipy.misc import imread
import cPickle as pickle
import pandas as pd
word2vec = gensim.models.Word2Vec
#glove_path='/home/flash/Documents/yash_papers/glove.6B_2/'
path = '/home/flash/Documents/yash_papers/20news/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

vocab_dim = 336026                         #2196018 for previous
dim = 50
batch_size = 16
n_epoch = 10
input_length = 50



# print('Indexing word vectors.')

# embeddings_index = {}
# f = open('/home/flash/Documents/yash_papers/glove.6B_2/glove.6B.300d.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()

# print('Found %s word vectors.' % len(embeddings_index))
model = gensim.models.Word2Vec.load_word2vec_format('/home/flash/Documents/yash_papers/glove.6B_2/glove.6B.50d.txt', binary=False)

print("Model loaded")


for dir_name in os.listdir(path):
    if os.path.isdir(path+dir_name):
        cnt = 0
        for fn in os.listdir(path+dir_name):
            cnt += 1
        print(dir_name, '->', cnt, 'files')


documents, labels = [], []
for dir_name in os.listdir(path):
    if os.path.isdir(path+dir_name):
        for fn in os.listdir(path+dir_name):
            with codecs.open(path+dir_name+'/'+fn, 'r', 'utf8') as f:
                try:
                    text = f.read()
                    documents.append(text)
                    labels.append(dir_name)
                except UnicodeDecodeError:
                    pass

print('# documents:', len(documents))
print('# labels:', len(labels))

#print("document1",documents[0])


def clean_data(data):
    lemmatizer=WordNetLemmatizer()
    word_list1=[]
    word_list = data.split()
    for i in range(len(word_list)):
        word_list1.append(word_list[i].strip('<,>,\n'))
    filtered_words=[word for word in word_list1 if word not in stopwords.words('english')]
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   filtered_words1=list(''.join(w for w in word_list1 if w not w.isdigit()))
    #repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    #repl = r'\1\2\3'
    mod_doc = []
    mod_doc1=[]
    for i in filtered_words:
        #i = unicode(i,errors='ignore')
        i.lower()
        i.strip('#\'"?,.!,')
        i.strip('<')
        if '@'  in i or 'http:' in i:
            continue
        j = re.sub(r'(.)\1+',r'\1\1',i)
        mod_doc.append(lemmatizer.lemmatize(j))
    #for k in mod_doc:
    #    mod_doc1.append(mod_doc[k].strip('\n'))
    return mod_doc
def tokenizer(text):
    text = [clean_data(document) for document in documents]
    print(text)
    return text
text_try=clean_data(documents[0])
text=[]


text = [clean_data(document) for document in documents]
# X_train=[]
# X_train=text[0:13000]
# X_test=text[13000:]
# Y_train=labels[0:13000]
# Y_test=labels[13000:]
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(text, labels,
                    test_size=0.10, random_state=42982)

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)


# for i in range(5):
#     text.append(clean_data(documents[i]))
# print(text)
# print(documents[0])
# print("---------------------------------preprocessed---------------------------------------")
# print(text_try)


# print(labels[0])


#print (model.most_similar('good'))


gensim_dict = Dictionary()
gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)
index_dict = {v: k+1 for k, v in gensim_dict.items()}
word_vectors = {word: model[word] for word in index_dict.keys()}


print('Setting up Arrays for Keras Embedding Layer...')
n_symbols = len(index_dict) + 1  # adding 1 to account for 0th index
embedding_weights = np.zeros((n_symbols, dim))
for word, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[word]
def sentence_to_vectors(data):

    max_len = 0
    transformed_train = []
    transformed_test = []
    
    for sent in data:
        #print (sent)
        #txt = sent.translate(None, string.punctuation)
        #txt = nltk.word_tokenize(str(sent).lower().replace("'s",'is'))   #More text processing later
        if len(sent) > max_len:
            max_len = len(sent)
        new_txt = []
        for word in sent:
            try:
                new_txt.append(index_dict[word])
            except:
                new_txt.append(0) # Vector of new word is set to 0
        transformed_train.append(new_txt)

    
    # for sent in test:
    #     #txt = sent.translate(None, string.punctuation)
    #     #txt = nltk.word_tokenize(str(sent).lower().replace("'s",'is'))   #More text processing later
    #     if len(txt) > max_len:
    #         max_len = len(txt)
    #     new_txt = []
    #     for word in txt:
    #         try:
    #             new_txt.append(index_dict[word])
    #         except:
    #             new_txt.append(0) # Vector of new word is set to 0
    #     transformed_test.append(new_txt)

    
    # print(len(transformed_train))
    # print(max_len)
    # print(len(transformed_test))
    
    return transformed_train,max_len



#SAVE LE NEXT TIME


features_train,max_len = sentence_to_vectors(text)
print(max_len)
data1=[]
data1 = pad_sequences(features_train, maxlen=max_len)

model1=Sequential()
model1.add(Embedding(output_dim=max_len,input_dim=vocab_dim,mask_zero=True,weights=[embedding_weights],input_length=input_length))

model1.add(LSTM(max_len))

model1.add(Dropout(0.25))

model1.add(Dense(2,actovation='sigmoid'))

model1.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model1.fit(X_train, Y_train, batch_size=8, nb_epoch=5,
          validation_data=(X_test, Y_test))

print("Evaluate...")
score, acc = model1.evaluate(X_test, Y_test,
                            batch_size=8)
print('Test score:', score)
print('Test accuracy:', acc)
