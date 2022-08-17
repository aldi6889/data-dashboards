import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openpyxl import Workbook
from sklearn import metrics

import csv
import preprocessor as p
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import seaborn as sns
import pandas as pd
import numpy as np
import random

def show_pie(label, data, legend_title) :
    fig, ax = plt.subplots(figsize=(8, 10), subplot_kw=dict(aspect='equal'))
    
    labels = [x.split()[-1] for x in label]
    
    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}% ({:d})".format(pct, absolute)
    
    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),textprops=dict(color="w"))
    ax.legend(wedges, labels,
            title= legend_title,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.setp(autotexts, size=10, weight="bold")
    st.pyplot(fig)

# import time


# st.title('Uber pickups in NYC')

# my_bar = st.progress(0)

# if st.button('Say hello'):
#   for percent_complete in range(100):
#      time.sleep(0.1)
#      my_bar.progress(percent_complete + 1)  
# else:
#      st.write('Goodbye')





#data1 = pd.read_csv('translate_lepasmasker.csv')

#data1.to_excel('translate_lepasmasker.xlsx',encoding='utf8', index=False)

# bagian menu BERANDA
def home():
    st.title('Beranda')
#end of home

# bagian menu Lexicon

def lex():
    st.title('Lexicon')

    data_org = pd.read_csv('bersih_lepasmasker.csv')
    data = pd.read_csv('translate_lepasmasker.csv')

    tweet_df = pd.read_csv('hasillepasmasker.csv')

    st.subheader('Tweet')
    st.dataframe(data_org.head())

    st.subheader('Tweet Translated')
    st.dataframe(data.head())

    st.subheader('Sentiments')
    s = pd.value_counts(tweet_df['Sentiments'])
    fig = plt.figure(figsize=(10, 4))
    ax = s.plot.bar()
    n = len(tweet_df.index)
    # print (n)
    for p in ax.patches:
        ax.annotate(str(round(p.get_height() / n * 100, 2)) + '%', (p.get_x() * 1.005, p.get_height() * 1.005))
    st.pyplot(fig)

    st.subheader('Word Cloud All Tweet')
    wordcloud1 = WordCloud(width = 800, height = 800, background_color = 'black', max_words = 1000 , min_font_size = 20).generate(str(tweet_df['Tweet']))
    #plot the word cloud
    fig = plt.figure(figsize = (8,8), facecolor = None)
    plt.imshow(wordcloud1)
    plt.axis('off')
    st.pyplot(fig)

    st.subheader('Word Cloud Positive Sentiment')
    all_words = ' '.join([text for text in tweet_df['Tweet'][tweet_df.Sentiments == "Positif"]])
    wordcloud2 = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5, colormap='Dark2',).generate(all_words) 
    fig = plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud2, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(fig)



    st.subheader('dataklasifikasi.xlsx')
    #masukan data label secara manual
    dataset = pd.read_excel('dataklasifikasi.xlsx')
    st.write(dataset.head())

    st.subheader('Compound Score')
    analyser = SentimentIntensityAnalyzer()
    scores = [analyser.polarity_scores(x) for x in dataset['Tweet']]
    dataset['Compound_Score'] = [x['compound'] for x in scores]
    st.write(dataset.head())

    st.subheader('Compound Score to Sentiment')

    dataset.loc[dataset['Compound_Score'] < 0, 'Sentiments'] = 'Negatif'
    dataset.loc[dataset['Compound_Score'] == 0, 'Sentiments'] = 'Netral'
    dataset.loc[dataset['Compound_Score'] > 0, 'Sentiments'] = 'Positif'

    st.write('Compound Score < 0,Sentimen Negatif')
    st.write('Compound Score == 0,Sentimen Netral')
    st.write('Compound Score > 0,Sentimen Positif')

    st.write(dataset.head())

    
    st.subheader('Metrics')

    st.dataframe(metrics.classification_report(dataset['Klasifikasi'], dataset['Sentiments'],output_dict=True))


#end of Lexicon

# bagian menu Naive Bayes
def nby():
    st.title('Naive Bayes')

    data_org    = pd.read_csv('bersih_lepasmasker.csv')
    data  = pd.read_csv('translate_lepasmasker.csv')

    st.subheader('Tweet')
    st.dataframe(data_org.head())

    st.subheader('Tweet Translated')
    st.dataframe(data.head())

    
    ps = PorterStemmer()

    def stemming_data(x):
        return ps.stem(x)

    data['Tweet'] = data['Tweet'].apply(stemming_data)   

    data_tweet = list(data['Tweet'])

    polaritas = 0

    status1 = []
    total_positif = total_negatif = total_netral = total = 0

    for i, tweet in enumerate(data_tweet):
        analysis = TextBlob(tweet)
        polaritas += analysis.polarity
        
        if analysis.sentiment.polarity > 0.0:
            total_positif += 1
            status1.append('Positif')
        elif analysis.sentiment.polarity == 0.0:
            total_netral += 1
            status1.append('Netral')
        else:
            total_negatif += 1
            status1.append('Negatif')
            
        total += 1

    data['klasifikasi'] = pd.DataFrame({'klasifikasi': status1})


    st.subheader('Tweet Sentiment Analyzed Using Textblob')

    st.dataframe(data.tail())
    label = ['Positif', 'Negatif', 'Netral']
    count_data = [total_positif+1, total_negatif+1, total_netral]

    show_pie(label, count_data, "Status")


    nltk.download('punkt')
    dataset = data.drop([], axis=1, inplace=False)
    dataset = [tuple(x) for x in dataset.to_records(index=False)]

    set_positif = []
    set_negatif = []
    set_netral = []

    for n in dataset:
        if(n[1] == 'Positif'):
            set_positif.append(n)
        elif(n[1] == 'Negatif'):
            set_negatif.append(n)
        else:
            set_netral.append(n)
            
    set_positif = random.sample(set_positif, k=int(len(set_positif)/2))
    set_negatif = random.sample(set_negatif, k=int(len(set_negatif)/2))
    set_netral = random.sample(set_netral, k=int(len(set_netral)/2))

    train = set_positif + set_negatif + set_netral

    train_set = []

    for n in train:
        train_set.append(n)


    cl = NaiveBayesClassifier(train_set)

    data_tweet = list(data['Tweet'])
    polaritas = 0

    # st.write(data_tweet)

    status = []
    total = total_positif = total_negatif = total_netral = 0    

    for i, tweet in enumerate(data_tweet):
        analysis = TextBlob(tweet, classifier=cl)
        
        if analysis.classify() == 'Positif':
            total_positif += 1
        elif analysis.classify() == 'Netral':
            total_netral += 1
        elif analysis.classify() == 'Negatif':
            total_negatif += 1
            
        status.append(analysis.classify())
        total += 1

    # st.write(f'\nHasil Analisis Data :\nPositif = {total_positif}\nNetral = {total_netral}\nNegatif = {total_negatif}')
    # st.write(f'\nTotal Data : {total}')

    data['klasifikasi_bayes'] = pd.DataFrame({'klasifikasi_bayes': status})

    st.subheader('Tweet Sentiment Analyzed Using Naive Bayes')
    st.dataframe(data.tail())
    label = ['Positif', 'Negatif', 'Netral']
    count_data = [total_positif+1, total_negatif+1, total_netral]
    show_pie(label, count_data, "Status")


    st.subheader('Word Cloud All Tweet')
    wordcloud1 = WordCloud(width = 800, height = 800, background_color = 'black', max_words = 1000 , min_font_size = 20).generate(str(data['Tweet']))
    #plot the word cloud
    fig = plt.figure(figsize = (8,8), facecolor = None)
    plt.imshow(wordcloud1)
    plt.axis('off')
    st.pyplot(fig)

    st.subheader('Word Cloud Positive Sentiment')
    all_words = ' '.join([text for text in data['Tweet'][data.klasifikasi_bayes == "Positif"]])
    wordcloud2 = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5, colormap='Dark2',).generate(all_words) 
    fig = plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud2, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(fig)


    st.subheader('Metrics')
    st.dataframe(metrics.classification_report(data['klasifikasi'], data['klasifikasi_bayes'],output_dict=True))


#end of Naive Bayes 


def main():
    activities = ['Beranda','Lexicon','Naive Bayes']

    st.sidebar.subheader("Menu")
    choice = st.sidebar.radio('',activities)
    if choice == 'Beranda':
        home()
    elif choice == 'Lexicon':
        lex()
    elif choice == 'Naive Bayes':
        nby()

if __name__ == '__main__':
    main()

