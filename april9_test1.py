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


from sklearn.utils import shuffle

import h5py
import pandas as pd
word2vec = gensim.models.Word2Vec
#glove_path='/home/flash/Documents/yash_papers/glove.6B_2/'
path = '/home/flash/Documents/yash_papers/20_newsgroup/'


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
print(documents[0])


def clean_data(data):
    lemmatizer=WordNetLemmatizer()
    word_list1=[]
    word_list = data.split()
    for i in range(len(word_list)):
        word_list1.append(word_list[i].strip('<,>,\n'))
    filtered_words=[word for word in word_list1 if word not in stopwords.words('english')]
    #print("Filtered Words>>>>")
    #print(filtered_words)
    #filtered_words1=list(''.join(w for w in word_list1 if w not w.isdigit()))
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
    mod_doc1=' '.join(mod_doc)
    return mod_doc1

#text = [clean_data(document) for document in documents]

new_documents=[]
for doc in documents:
    new_documents.append(clean_data(doc))

#print(new_documents[0])

from sklearn.feature_extraction.text import TfidfVectorizer

'''
vectorizer = TfidfVectorizer(max_features=5000, analyzer='word', use_idf=False)
X = vectorizer.fit_transform(new_documents).toarray() # unsparsify!
print(X.shape)
import pickle
#pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print(y.shape)
print(y[:10])
print(y[-10:])


from keras.utils import np_utils
Y = np_utils.to_categorical(y)
print(Y.shape)
print(Y[:5])
print(Y[-5:])



from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                    test_size=0.10, random_state=42982)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train,
                     test_size=0.10, random_state=476710)

print('Train data:', X_train.shape)
print('Dev data:', X_dev.shape)
print('Test data:', X_test.shape)



from keras.models import Sequential
from keras.layers.core import Dense
#model = Sequential()
nb_classes = Y.shape[1]
nb_features = X.shape[1]


'
from keras.layers.core import Dense


nb_classes = Y.shape[1]
nb_features = X.shape[1]

model.add(Dense(nb_classes, input_dim=nb_features, activation='softmax'))

json_string = model.to_json()
open('my_model_architecture.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5', overwrite=True)

'''

from keras.models import model_from_json
model = model_from_json(open('model.json').read())
model.load_weights('model.h5')




from keras.layers.core import Activation
'''
model = Sequential()
#model.add(Conv1D(input_dim=5000,nb_filters=128,filter_length=5))
model.add(Dense(5000,input_dim=5000))
model.add(Activation('relu'))
model.add(Dense(2500))
model.add(Activation('sigmoid'))
model.add(Dense(1000))
model.add(Activation('sigmoid'))
model.add(Dense(250))

#model.add(Dense(128),input_dim=nb_features)
# model.add(Activation('relu'))
# model.add(Dense(128))
# model.add(Activation('relu'))
model.add(Dense(nb_classes, input_dim=nb_features, name='dense1'))
model.add(Activation('softmax', name='softmax1'))
model.summary()
'''

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


#model.fit(X_train, Y_train, batch_size=50, nb_epoch=25)
#model.save('yash_1.h5')

#model.fit(X_train, Y_train, batch_size=50, nb_epoch=25, validation_data=(X_dev, Y_dev))
#model=load_model('yash.h5')
#preds = model.predict_classes(X_test)
#print(preds[:10])

### ###################################################################################


#preds = model.predict(X_test)
#class_ints = preds.argmax(axis=1)
#print(class_ints[:10])

import pickle
vectorizer=pickle.load(open(("feature.pkl"),'rb'))
label_encoder=pickle.load(open(("encoder.pkl"),'rb'))
X = vectorizer.transform(new_documents).toarray() # unsparsify!

print(X.shape)

y = label_encoder.transform(labels)
print(y.shape)
print(y[:10])
print(y[-10:])


from keras.utils import np_utils
Y = np_utils.to_categorical(y)

print(Y.shape)

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                    test_size=0.10, random_state=42982)

#X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train,
 #                    test_size=0.10, random_state=476710)

print('Train data:', X_train.shape)
print('Test data:', X_test.shape)


#predicted_classes = label_encoder.inverse_transform(class_ints)
#correct_classes = label_encoder.inverse_transform(Y_test.argmax(axis=1))
#for i, j in zip(predicted_classes, correct_classes)[:10]:
 #   print(i, '->', j)

preds = model.predict_classes(X_test)
print(preds[:10])

######################################################################################


preds = model.predict(X_test)
class_ints = preds.argmax(axis=1)
print(class_ints[:10])

predicted_classes = label_encoder.inverse_transform(class_ints)
correct_classes = label_encoder.inverse_transform(Y_test.argmax(axis=1))
for i, j in zip(predicted_classes, correct_classes)[:10]:
    print(i, '->', j)

from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import precision_recall_fscore_support
print('Test accuracy:', accuracy_score(predicted_classes, correct_classes))

print('Test precision:', precision_score(predicted_classes, correct_classes))

print('Test recall:', recall_score(predicted_classes, correct_classes))

print('Test f1_score:', f1_score(predicted_classes, correct_classes))
precision,recall,fscore,support=precision_recall_fscore_support(predicted_classes, correct_classes)



# precision_1=precision.argmax(axis=1)
# recall_1=recall.argmax(axis=1)
# fscore_1=fscore.argmax(axis=1)
# support_1=support.argmax(axis=1)

print("precision:",precision)
print("recall:",recall)
print("fscore:",fscore)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("Classification report:")
target_names=['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.os.ms-windows.misc','comp.sys.mac.hardware','comp.sys.mac.hardware','misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']
print(classification_report(predicted_classes,correct_classes,target_names=target_names))
#print('Test accuracy:', accuracy_score(predicted_classes, correct_classes))

#new_file="baghu"
print("Enter File name (.txt)")
new_file=raw_input()

documents1=[]
with codecs.open(new_file, 'r', 'utf8') as f:
	text1=f.read()

	documents1.append(text1)
new_doc=[]
new_doc.append(clean_data(text1))


X1 = vectorizer.transform(new_doc).toarray() 
print(X1)
preds=model.predict(X1)
print(model.predict(X1))
print(model.predict_classes(X1))

class_ints = preds.argmax(axis=1)
predicted_classes = label_encoder.inverse_transform(class_ints)
# correct_classes = label_encoder.inverse_transform(Y_test.argmax(axis=1))
# for i, j in zip(predicted_classes, correct_classes)[:10]:
#     print(i, '->', j)
print(predicted_classes)