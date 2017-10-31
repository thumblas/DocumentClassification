import re
import xml.sax.saxutils as saxutils

from BeautifulSoup import BeautifulSoup

from gensim.models.word2vec import Word2Vec

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM

from multiprocessing import cpu_count

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer

from pandas import DataFrame

from sklearn.cross_validation import train_test_split
import random
import numpy as np

random.seed(1000)

data_folder = '/home/flash/Documents/yash_papers/reuters21578/'

sgml_number_of_files = 22

sgml_file_name_template = 'reut2-NNN.sgm'

category_files = {
    'to_': ('Topics', 'all-topics-strings.lc.txt'),
    'pl_': ('Places', 'all-places-strings.lc.txt'),
    'pe_': ('People', 'all-people-strings.lc.txt'),
    'or_': ('Organizations', 'all-orgs-strings.lc.txt'),
    'ex_': ('Exchanges', 'all-exchanges-strings.lc.txt')
}

num_features = 500

document_max_num_words = 100

selected_categories = ['pl_usa']

category_data = []

for category_prefix in category_files.keys():
	with open(data_folder + category_files[category_prefix][1], 'r') as file:
		for category in file.readlines():
			category_data.append([category_prefix + category.strip().lower(),category_files[category_prefix][0],0])

	news_categories = DataFrame(data=category_data, columns=['Name', 'Type', 'Newslines'])


def update_frequencies(categories):
	for category in categories:
		idx = news_categories[news_categories.Name == category].index[0]
		f = news_categories.get_value(idx, 'Newslines')
		news_categories.set_value(idx, 'Newslines', f+1)

def to_category_vector(categories, target_categories):
	vector = np.zeros(len(target_categories)).astype(np.float32)

	for i in range(len(target_categories)):
		if target_categories[i] in categories:
			vector[i] = 1.0


	return vector

document_X = {}
document_Y = {}

def strip_tags(text):
	return re.sub('<[^<]+?>', '', text).strip()

def unescape(text):
	return saxutils.unescape(text)

for i in range(sgml_number_of_files):
    if i < 10:
        seq = '00' + str(i)
    else:
        seq = '0' + str(i)
        
    file_name = sgml_file_name_template.replace('NNN', seq)
    print('Reading file: %s' % file_name)
    
    with open(data_folder + file_name, 'r') as file:
        content = BeautifulSoup(file.read().lower())
        
        for newsline in content('reuters'):
            document_categories = []
            
            # News-line Id
            document_id = newsline['newid']
            
            # News-line text
            document_body = strip_tags(str(newsline('text')[0].body)).replace('reuter\n&#3;', '')
            document_body = unescape(document_body)
            
            # News-line categories
            topics = newsline.topics.contents
            places = newsline.places.contents
            people = newsline.people.contents
            orgs = newsline.orgs.contents
            exchanges = newsline.exchanges.contents
            
            for topic in topics:
                document_categories.append('to_' + strip_tags(str(topic)))
                
            for place in places:
                document_categories.append('pl_' + strip_tags(str(place)))
                
            for person in people:
                document_categories.append('pe_' + strip_tags(str(person)))
                
            for org in orgs:
                document_categories.append('or_' + strip_tags(str(org)))
                
            for exchange in exchanges:
                document_categories.append('ex_' + strip_tags(str(exchange)))
                
            # Create new document    
            update_frequencies(document_categories)
            
            document_X[document_id] = document_body
            document_Y[document_id] = to_category_vector(document_categories, selected_categories)


news_categories.sort_values(by='Newslines', ascending=False, inplace=True)
print(news_categories.head(20))



stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer('[\'a-zA-Z]+')

lemmatizer = WordNetLemmatizer()

newsline_documents = []

def tokenize(document):
	words = []

	for sentence in sent_tokenize(document):
		tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(sentence) if t.lower() not in stop_words]
		words += tokens

	return words

for key in document_X.keys():
	newsline_documents.append(tokenize(document_X[key]))

number_of_documents = len(document_X)

w2v_model = Word2Vec.load(data_folder + 'reuters.word2vec')

'''w2v_model = Word2Vec(newsline_documents, size=num_features, min_count=1, window=10, workers=cpu_count())
w2v_model.init_sims(replace=True)
w2v_model.save(data_folder + 'reuters.word2vec')

'''
print(selected_categories)
num_categories = len(selected_categories)
X = np.zeros(shape=(number_of_documents, document_max_num_words, num_features)).astype(np.float32)
Y = np.zeros(shape=(number_of_documents, num_categories)).astype(np.float32)

empty_word = np.zeros(num_features).astype(np.float32)

for idx, document in enumerate(newsline_documents):
	for jdx, word in enumerate(document):
		if jdx == document_max_num_words:
			break
            
		else:
			if word in w2v_model:
				X[idx, jdx, :] = w2v_model[word]
			else:
				X[idx, jdx, :] = empty_word

for idx, key in enumerate(document_Y.keys()):
	Y[idx, :] = document_Y[key]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = Sequential()

model.add(LSTM(int(document_max_num_words*1.5), input_shape=(document_max_num_words, num_features)))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Dense(num_categories))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(X_train, Y_train, batch_size=128, nb_epoch=5, validation_data=(X_test, Y_test))
#model.save('feb18.h5')
#model.save_weights('feb18_weights.h5')
# Evaluate model
model.load_weights('feb18_weights.h5')
score, acc = model.evaluate(X_test, Y_test, batch_size=128)
    
print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)

print(model.predict_classes(X_test[1:5]))




