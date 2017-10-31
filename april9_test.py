def solve(new_file):

    #from __future__ import print_function
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


    from sklearn.utils import shuffle

    import h5py
    import pandas as pd
    word2vec = gensim.models.Word2Vec

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


    from keras.models import model_from_json
    model = model_from_json(open('model.json').read())
    model.load_weights('model.h5')




    from keras.layers.core import Activation

    #model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


    import pickle
    vectorizer=pickle.load(open(("feature.pkl"),'rb'))
    label_encoder=pickle.load(open(("encoder.pkl"),'rb'))


    from sklearn.metrics import accuracy_score
    # print("Enter File name (.txt)")
    # new_file=raw_input()

    documents1=[]
    with codecs.open(new_file, 'r', 'utf8') as f:
    	text1=f.read()

    	documents1.append(text1)
    new_doc=[]
    new_doc.append(clean_data(text1))


    X1 = vectorizer.transform(new_doc).toarray() 
    #print(X1)
    preds=model.predict(X1)
    #print(model.predict(X1))
    #print(model.predict_classes(X1))

    class_ints = preds.argmax(axis=1)
    predicted_classes = label_encoder.inverse_transform(class_ints)
    #print(predicted_classes)
    return predicted_classes


print("File name to classify:>>>>>")

print(solve(raw_input()))