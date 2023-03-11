import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplt
import matplotlib as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re 
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
import pickle

training_dataset = pd.read_csv("twitter_training.csv",header=None)
validation_dataset = pd.read_csv("twitter_validation.csv",header=None)


print (training_dataset.head())
print(training_dataset.columns)

print (validation_dataset.head())
print(validation_dataset.columns)

training_dataset.columns = ['id','information','type','text']
validation_dataset.columns = ['id','information','type','text']

print(training_dataset.head())
print(validation_dataset.head())


training_dataset['lower'] = training_dataset.text.str.lower()
training_dataset['lower'] = [str(data) for data in training_dataset.lower]
training_dataset['lower'] = training_dataset.lower.apply(lambda x: re.sub('[^ A-Za-z0-9 ]+',' ',x))

validation_dataset['lower'] = validation_dataset.text.str.lower()
validation_dataset['lower'] = [str(data) for data in validation_dataset.lower]
validation_dataset['lower'] = validation_dataset.lower.apply(lambda x: re.sub('[^ A-Za-z0-9 ]+',' ',x))

print(training_dataset.head())
print(validation_dataset.head())


word_cloud_text = ''.join(training_dataset[training_dataset['type']=='Positive'].lower)

wordcloud = WordCloud(
    max_font_size = 100,
    max_words = 100,
    background_color = 'navy',
    scale = 10,
    width = 800,
    height = 800
).generate(word_cloud_text)

pyplt.figure(figsize=(10,10))
pyplt.imshow(wordcloud,interpolation = 'bilinear')
pyplt.axis('off')
pyplt.show()
wordcloud.to_file("wordcloud_positive.png")


word_cloud_text = ''.join(training_dataset[training_dataset['type']=='Negative'].lower)

wordcloud = WordCloud(
    max_font_size = 100,
    max_words = 100,
    background_color = 'navy',
    scale = 10,
    width = 800,
    height = 800
).generate(word_cloud_text)

pyplt.figure(figsize=(10,10))
pyplt.imshow(wordcloud,interpolation = 'bilinear')
pyplt.axis('off')
pyplt.show()
wordcloud.to_file("wordcloud_negative.png")

word_cloud_text = ''.join(training_dataset[training_dataset['type']=='Irrelevant'].lower)

wordcloud = WordCloud(
    max_font_size = 100,
    max_words = 100,
    background_color = 'navy',
    scale = 10,
    width = 800,
    height = 800
).generate(word_cloud_text)

pyplt.figure(figsize=(10,10))
pyplt.imshow(wordcloud,interpolation = 'bilinear')
pyplt.axis('off')
pyplt.show()
wordcloud.to_file("wordcloud_irrelevant.png")

word_cloud_text = ''.join(training_dataset[training_dataset['type']=='Neutral'].lower)

wordcloud = WordCloud(
    max_font_size = 100,
    max_words = 100,
    background_color = 'navy',
    scale = 10,
    width = 800,
    height = 800
).generate(word_cloud_text)

pyplt.figure(figsize=(10,10))
pyplt.imshow(wordcloud,interpolation = 'bilinear')
pyplt.axis('off')
pyplt.show()
wordcloud.to_file("wordcloud_neutral.png")


plot1 = training_dataset.groupby(by=['information','type']).count().reset_index()
print(plot1.head())
pyplt.figure(figsize=(20,10))
sns.barplot(data=plot1,x='information',y='id',hue='type')
pyplt.xticks (rotation = 90)
pyplt.xlabel("Brand")
pyplt.ylabel("Number of Tweets")
pyplt.grid()
pyplt.title("Distribution of Tweets per Brand and Type")
pyplt.show()
pyplt.savefig("Distribution of Tweets per Brand and Type")

tokens_text = [word_tokenize(str(word)) for word in training_dataset.lower]
tokens_counter = [item for sublist in tokens_text for item in sublist]
print("Number of Tokens:  ",len(set(tokens_counter)))

print(tokens_text[1])

stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')
print(stop_words[:5])

bow_counts = CountVectorizer(
    tokenizer = word_tokenize,
    stop_words = stop_words,
    ngram_range=(1,1)
)

reviews_train, reviews_test = train_test_split(training_dataset, test_size=0.2, random_state=0) 

X_train_bow = bow_counts.fit_transform(reviews_train.lower)
X_test_bow = bow_counts.transform(reviews_test.lower)

y_train_bow = reviews_train['type']
y_test_bow = reviews_test['type']

print(y_test_bow.value_counts()/y_test_bow.shape[0])

modelL1 = LogisticRegression(C=1, solver = "liblinear", max_iter = 400)
modelL1.fit(X_train_bow, y_train_bow)
test_pred = modelL1.predict(X_test_bow)
print("Accuracy:  ",accuracy_score(y_test_bow,test_pred)*100)

X_val_bow = bow_counts.transform(validation_dataset.lower)
y_val_bow = validation_dataset['type']

validation_dataset_result = modelL1.predict(X_val_bow)
print("Validation:  ",accuracy_score(y_val_bow,validation_dataset_result)*100)

bow_counts = CountVectorizer(
    tokenizer = word_tokenize,
    ngram_range=(1,4)
)

X_train_bow = bow_counts.fit_transform(reviews_train.lower)
X_test_bow = bow_counts.transform(reviews_test.lower)
X_val_bow = bow_counts.transform(validation_dataset.lower)

modelL2 = LogisticRegression(C=0.9, solver = "liblinear", max_iter = 400)
modelL2.fit(X_train_bow, y_train_bow)
test_pred = modelL2.predict(X_test_bow)
print("Accuracy:  ",accuracy_score(y_test_bow,test_pred)*100)

y_val_bow = validation_dataset['type']
validation_dataset_result = modelL2.predict(X_val_bow)
print("Validation Accuracy:  ",accuracy_score(y_val_bow,validation_dataset_result)*100)

filename = 'Logisitic_regression_model_01.sav'
pickle.dump(modelL1, open(filename, 'wb'))

filename = 'Logisitic_regression_model_02(optimized).sav'
pickle.dump(modelL2, open(filename, 'wb'))