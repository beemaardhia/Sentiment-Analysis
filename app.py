
import nltk

# Membuat fungsi untuk memeriksa apakah koleksi data sudah terinstal
def check_and_download(collection_name):
    try:
        nltk.data.find(collection_name)
    except LookupError:
        print(f"{collection_name} belum terinstal. Mengunduh koleksi data...")
        nltk.download(collection_name)
        print(f"{collection_name} berhasil diunduh.")
    else:
        print(f"{collection_name} sudah terinstal.")

# Memeriksa dan mengunduh koleksi data 'punkt'
check_and_download('punkt')

# Memeriksa dan mengunduh koleksi data 'stopwords'
check_and_download('stopwords')

import streamlit as st
from textblob import TextBlob
import pandas as pd
import re
import string
from library import KNN, TFIDF
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords



def case_fold(text):
    text = text.lower()
    text = re.sub('\[.*?\]',' ',text)
    text = re.sub('https?://\S+|www\.\S+',' ',text)
    text = re.sub('<.*/>+',' ',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),' ',text)
    text = re.sub('\n',' ',text)
    text = re.sub('\w*\d\w*',' ',text)
    text = re.sub('[^a-z]',' ',text)
    text = re.sub(r'(.)\1{2,}',r'\1',text)
    return text
        

key_norm = pd.read_csv('key_norm_indo.csv')

def text_norm(text):
    text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in text.split()])
    text = text.lower()
    return text


stopword_ind = stopwords.words('indonesian')

def remove_stopword(text):
    clean=[]
    text = text.split()
    
    for word in text:
        if word not in stopword_ind:
            clean.append(word)
    return ' '.join(clean)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming (text):
    text = stemmer.stem(text)
    return text

def preprocess (test):
    test = case_fold(test)
    test = text_norm(test) 
    test = remove_stopword(test)
    test = stemming(test)
    return test

    
X = pd.read_csv('X_trainraw.csv')
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

X = X['text']
X_train.drop(['Unnamed: 0'], axis=1, inplace=True)
y_train.drop(['Unnamed: 0'], axis=1, inplace=True)

X = X.to_numpy().ravel()
y_train = y_train.to_numpy().ravel()
X_train =X_train.to_numpy()

tfii = TFIDF()
tfii.fit(X)

model = KNN()
model.fit(X_train,y_train)

def predict (text):
    cleaned = preprocess(text)
    data_tf = tfii.transform([cleaned])
    pred = model.predict(data_tf)
    return pred

def preproced_eng(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    blob = TextBlob(text)
    correct = blob.correct()
    words = correct.words
    lemmatized = ' '.join(word.lemmatize() for word in words)
    return lemmatized.lower()


def get_sentimen(text):
    blob = TextBlob(text)
    pola = blob.polarity
    if pola > 0.1:
        sent='Positive'
    elif pola < -0.1:
        sent='Negative'
    else:
        sent='Netral'
    return sent


st.header('Analisis Sentimen')

with st.expander('Analisi Teks Bahasa Indonesia'):
    text_indo = st.text_area('Tulis di sini:')
    
    if st.button('Analyze'):
        if text_indo:
            sentiment = predict(text_indo)
            st.write('Hasil Analisis : ', sentiment[0])
            
with st.expander('Analisis .csv Bahasa Indonesia'):
    upl = st.file_uploader('Upload .csv file ( nama kolom yang diprediksi harus "text" )', type='csv')
    
    if upl:
        data = pd.read_csv(upl, on_bad_lines='skip')
        data['preproceed'] = data['text'].apply(preprocess)
        data['sentiment'] = data['preproceed'].apply(predict)
        
        st.write(data)
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        
        csv = convert_df(data)
        
        st.download_button(
            label='Download file as CSV',
            data=csv,
            file_name='predict.csv',
            mime='text/csv',
        )
            
        
            
    
st.write('---')
st.header('Sentiment Analysis')


with st.expander('English Analysis'):
    text_eng = st.text_area('Text here:')
    
    if st.button('Analyze', key='eng'):
        if text_eng:
            preprocedd = preproced_eng(text_eng)
            sent = get_sentimen(preprocedd)
            st.write('Pediction : ', sent)
            
with st.expander('English .csv Analysis'):
    upl = st.file_uploader('Upload .csv file (the predicted column name must be "text" )', type='csv', key='upl_eng')
    
    if upl:
        data = pd.read_csv(upl, on_bad_lines='skip')
        data['preproceed'] = data['text'].apply(preprocedd)
        data['sentiment'] = data['preproceed'].apply(get_sentimen)
        st.write(data)
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        
        csv = convert_df(data)
        
        st.download_button(
            label='Download file as CSV',
            data=csv,
            file_name='predict.csv',
            mime='text/csv'
        )
