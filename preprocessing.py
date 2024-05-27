# -*- coding: utf-8 -*-
"""
Created on Sat May 25 15:50:30 2024

@author: emirh
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer

# Gerekli NLTK bileşenlerini indiriyoruz
nltk.download('punkt')
nltk.download('stopwords')

# Metni kelimelere ayırma işlemi
def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

# Yaygın kelimeleri çıkarma işlemi
def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

# Normalizasyon işlemi: küçük harfe çevirme ve noktalama işaretlerini çıkarma
def normalize(tokens):
    normalized_tokens = [word.lower() for word in tokens if word.isalnum()]
    return normalized_tokens

# Kök bulma işlemi: Porter veya Snowball stemmer kullanma
def stem(tokens, method='porter'):
    if method == 'porter':
        stemmer = PorterStemmer()
    elif method == 'snowball':
        stemmer = SnowballStemmer('english')
    else:
        raise ValueError("method should be 'porter' or 'snowball'")
    
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens

# Tüm ön işleme adımlarını bir araya getiren fonksiyon
def preprocess_text(text, method='porter'):
    tokens = tokenize(text)  # Kelimelere ayırma
    tokens = remove_stop_words(tokens)  # Yaygın kelimeleri çıkarma
    tokens = normalize(tokens)  # Normalizasyon
    tokens = stem(tokens, method)  # Kök bulma
    return tokens

# Örnek bir metin üzerinde ön işleme adımlarını test ediyoruz
sample_text = "This is an example text to demonstrate the preprocessing steps, including tokenization, stop words removal, and stemming."
processed_text = preprocess_text(sample_text, method='porter')
print("Ön işlemden geçmiş örnek metin:", processed_text)

# cran.all dosyasını okuma fonksiyonu
def read_cran_all(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        data = file.read()
    
    documents = data.split('.I ')  # Dokümanları ayırma
    docs = []
    
    for doc in documents[1:]:  # İlk eleman boş olacağı için atlıyoruz
        doc_dict = {}
        lines = doc.split('\n')
        doc_dict['ID'] = lines[0].strip()
        doc_dict['Title'] = ""
        doc_dict['Author'] = ""
        doc_dict['Book'] = ""
        doc_dict['Words'] = ""
        for i in range(1, len(lines)):
            if lines[i].startswith('.T'):
                j = i + 1
                while j < len(lines) and not lines[j].startswith('.A'):
                    doc_dict['Title'] += lines[j] + " "
                    j += 1
            if lines[i].startswith('.A'):
                j = i + 1
                while j < len(lines) and not lines[j].startswith('.B'):
                    doc_dict['Author'] += lines[j] + " "
                    j += 1
            if lines[i].startswith('.B'):
                j = i + 1
                while j < len(lines) and not lines[j].startswith('.W'):
                    doc_dict['Book'] += lines[j] + " "
                    j += 1
            if lines[i].startswith('.W'):
                j = i + 1
                while j < len(lines):
                    doc_dict['Words'] += lines[j] + " "
                    j += 1
        docs.append(doc_dict)
    
    return docs

# cran.all dosyasını okuma ve ön işleme adımlarını uygulama
docs = read_cran_all('C:\\Users\\emirh\\Desktop\\bilgi_erisim\\cran.all.1400')
if docs:
    print("İlk doküman:")
    print(docs[0])  # İlk dokümanı yazdır

    # Dokümanları ön işlemden geçirme
    for doc in docs:
        doc['Processed_Words'] = preprocess_text(doc['Words'], method='porter')
    print("\nİlk dokümanın ön işlenmiş hali:")
    print(docs[0])  # İlk dokümanın ön işlenmiş halini yazdır
else:
    print("Dokümanlar yüklenemedi.")

# cran.qry dosyasını okuma fonksiyonu
def read_cran_qry(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        data = file.read()
    
    queries = data.split('.I ')  # Sorguları ayırma
    qry = []
    
    for query in queries[1:]:  # İlk eleman boş olacağı için atlıyoruz
        query_dict = {}
        lines = query.split('\n')
        query_dict['ID'] = lines[0].strip()
        query_dict['Words'] = ""
        for i in range(1, len(lines)):
            if lines[i].startswith('.W'):
                j = i + 1
                while j < len(lines):
                    query_dict['Words'] += lines[j] + " "
                    j += 1
        qry.append(query_dict)
    
    return qry

# cran.qry dosyasını okuma ve ön işleme adımlarını uygulama
queries = read_cran_qry('C:\\Users\\emirh\\Desktop\\bilgi_erisim\\cran.qry')
if queries:
    print("\nİlk sorgu:")
    print(queries[0])  # İlk sorguyu yazdır

    # Sorguları ön işlemden geçirme
    for query in queries:
        query['Processed_Words'] = preprocess_text(query['Words'], method='porter')
    print("\nİlk sorgunun ön işlenmiş hali:")
    print(queries[0])  # İlk sorgunun ön işlenmiş halini yazdır
else:
    print("Sorgular yüklenemedi.")
