from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
import streamlit as st
import gensim.downloader as api
nltk.download('punkt')
nltk.download('punkt_tab')


sample_text = """I love Machine Learning.
Machine Learning is fun.
Deep Learning is powerful.
I enjoy learning new things."""

text = st.text_area(
    "Enter the text",
    value=sample_text,
    height=200
)
wordtokens=word_tokenize(text)
senttokens=sent_tokenize(text)
col1,col2,col3,col4 =st.columns(4)

with col1:
    ohc=st.button("OHE")
with col2:
    bow=st.button("BOW")
with col3:
    tfidf=st.button("TF-IDF")
with col4:
    word2vec=st.button("Word2Vec")

if ohc:
    vectorizer=CountVectorizer(binary=True)
    st.write("Sentances:",senttokens)
    x=vectorizer.fit_transform(senttokens)
    st.write("vocabulary :",vectorizer.get_feature_names_out())
    st.write("one hot encoded matrix:",x.toarray())
elif bow:
    vectorizer=CountVectorizer()
    x=vectorizer.fit_transform(senttokens)
    st.write("vocabulary :",vectorizer.get_feature_names_out())
    st.write("One Hot Encoded Matrix :",x.toarray())
elif tfidf:
    vectorizer=TfidfVectorizer()
    tfidfmatrix=vectorizer.fit_transform(senttokens)
    st.write("\nIDF values :")
    for a,b in zip(vectorizer.get_feature_names_out(),vectorizer.idf_):
        st.write(a,b)
    st.write("\nfeature names:",vectorizer.get_feature_names_out())
    st.write("\nword indexes :",vectorizer.vocabulary_)
    st.write("\nTF-IDF Matrix:\n",tfidfmatrix)
    st.write("Matrix:",tfidfmatrix.toarray())
elif word2vec:
    model=api.load("word2vec-google-news-300")
    st.write("Word2Vec :")
    vectors = {}
    missing_words = []
    for word in wordtokens:
        if word in model:
            vectors[word] = model[word][:10]   # show only first 10 values
        else:
            missing_words.append(word)
    if vectors:
        st.write("Word Vectors (first 10 values of 300D):")
        for word, vec in vectors.items():
            st.write(f"{word} → {vec}")

    if missing_words:
        st.write("Words not found in Word2Vec vocabulary:")
        st.write(list(set(missing_words)))
