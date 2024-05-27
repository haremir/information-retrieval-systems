# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:30:30 2024

@author: emirh
"""

import json
import math
from preprocessing import preprocess_text, read_cran_all  # Fonksiyonları import ediyoruz

# Ters indeks yapısını oluşturma fonksiyonu
def create_inverted_index(docs):
    inverted_index = {}
    
    for doc in docs:
        doc_id = doc['ID']
        words = doc['Processed_Words']
        
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = {'doc_freq': 0, 'postings': {}}
            
            if doc_id not in inverted_index[word]['postings']:
                inverted_index[word]['postings'][doc_id] = 0
                inverted_index[word]['doc_freq'] += 1
            
            inverted_index[word]['postings'][doc_id] += 1
    
    # Toplam geçiş sayısını hesaplama
    for word in inverted_index:
        inverted_index[word]['total_term_freq'] = sum(inverted_index[word]['postings'].values())
    
    return inverted_index

# JSON olarak ters indeks yapısını kaydetme fonksiyonu
def save_inverted_index(inverted_index, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(inverted_index, file, ensure_ascii=False, indent=4)

# JSON dosyasını okuma ve görüntüleme fonksiyonu
def load_inverted_index(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        inverted_index = json.load(file)
    return inverted_index

# İki terimin posting listesini kesiştirme fonksiyonu
def intersect_postings(postings1, postings2):
    result = []
    i, j = 0, 0
    
    postings1 = sorted(postings1)
    postings2 = sorted(postings2)
    
    while i < len(postings1) and j < len(postings2):
        if postings1[i] == postings2[j]:
            result.append(postings1[i])
            i += 1
            j += 1
        elif postings1[i] < postings2[j]:
            i += 1
        else:
            j += 1
    
    return result

# Sorgu işleme fonksiyonu
def process_query(query, inverted_index):
    processed_query = preprocess_text(query, method='porter')
    if not processed_query:
        return []

    # Terimlerin doküman frekanslarına göre sıralanması
    term_postings = [(term, inverted_index[term]['postings']) for term in processed_query if term in inverted_index]
    term_postings.sort(key=lambda x: len(x[1]))

    if not term_postings:
        return []

    # İlk terimin posting listesi ile başlayarak kesişimi gerçekleştir
    result = list(term_postings[0][1].keys())
    for term, postings in term_postings[1:]:
        result = intersect_postings(result, postings.keys())
        if not result:
            break

    return result

# OR sorgusu işleme fonksiyonu
def process_or_query(query, inverted_index):
    processed_query = preprocess_text(query, method='porter')
    if not processed_query:
        return []

    result_set = set()
    for term in processed_query:
        if term in inverted_index:
            result_set.update(inverted_index[term]['postings'].keys())

    return list(result_set)


# TF-IDF hesaplama fonksiyonları
def compute_tf(doc):
    tf_dict = {}
    for word in doc:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / len(doc)
    return tf_dict

def compute_idf(docs):
    N = len(docs)
    idf_dict = {}
    for doc in docs:
        for word in doc:
            if word not in idf_dict:
                idf_dict[word] = 0
            idf_dict[word] += 1
    for word in idf_dict:
        idf_dict[word] = math.log(N / float(idf_dict[word]))
    return idf_dict

def compute_tfidf(tf, idf):
    tfidf = {}
    for word in tf:
        tfidf[word] = tf[word] * idf.get(word, 0)
    return tfidf

def calculate_tfidf_for_all_docs(docs):
    tfidf_docs = []
    docs_tf = [compute_tf(doc) for doc in docs]
    idf_dict = compute_idf(docs_tf)
    for tf in docs_tf:
        tfidf = compute_tfidf(tf, idf_dict)
        tfidf_docs.append(tfidf)
    return tfidf_docs

# Değerlendirme metrikleri hesaplama fonksiyonları
def calculate_precision_recall_f1(query_id, results, ground_truth):
    true_docs = ground_truth.get(query_id, [])
    retrieved_docs = results

    tp = len(set(retrieved_docs) & set(true_docs))
    fp = len(set(retrieved_docs) - set(true_docs))
    fn = len(set(true_docs) - set(retrieved_docs))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def calculate_map(query_results, ground_truth):
    ap_sum = 0
    for query_id, results in query_results.items():
        true_docs = ground_truth.get(query_id, [])
        retrieved_docs = results

        relevant_retrieved = 0
        precision_sum = 0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in true_docs:
                relevant_retrieved += 1
                precision_sum += relevant_retrieved / (i + 1)
        
        ap = precision_sum / len(true_docs) if true_docs else 0
        ap_sum += ap
    
    map_score = ap_sum / len(query_results) if query_results else 0
    return map_score

def evaluate_system(queries, ground_truth, inverted_index):
    query_results = {query["ID"]: process_query(query["Words"], inverted_index) for query in queries}

    # Precision, Recall, F1 Score hesapla
    metrics = {}
    for query_id in query_results:
        precision, recall, f1 = calculate_precision_recall_f1(query_id, query_results[query_id], ground_truth)
        metrics[query_id] = {"Precision": precision, "Recall": recall, "F1": f1}

    # MAP hesapla
    map_score = calculate_map(query_results, ground_truth)

    # Sonuçları yazdır
    for query_id, metric in metrics.items():
        print(f"Query ID: {query_id} - Precision: {metric['Precision']:.4f}, Recall: {metric['Recall']:.4f}, F1 Score: {metric['F1']:.4f}")

    print(f"Mean Average Precision (MAP): {map_score:.4f}")

# Sonuçları görüntüleme fonksiyonu
def display_results(relevant_docs, docs):
    if relevant_docs:
        print("Eşleşen Belgeler:\n")
        for doc_id in relevant_docs:
            doc = next((d for d in docs if d['ID'] == doc_id), None)
            if doc:
                print(f"ID: {doc['ID']}\nTitle: {doc['Title']}\nAuthor: {doc['Author']}\nBook: {doc['Book']}\n")
    else:
        print("Hiçbir eşleşen belge bulunamadı.")

# Ana çalışma fonksiyonu
def main():
    # Dosya yolları
    docs_path = 'C:\\Users\\emirh\\Desktop\\bilgi_erisim\\cran.all.1400'
    inverted_index_path = 'C:\\Users\\emirh\\Desktop\\bilgi_erisim\\inverted_index.json'
    
    # Dokümanları yükleme
    docs = read_cran_all(docs_path)
    
    if docs:
        # Dokümanları ön işlemden geçirme
        for doc in docs:
            doc['Processed_Words'] = preprocess_text(doc['Words'], method='porter')
        
        # Ters indeks yapısını oluşturma
        inverted_index = create_inverted_index(docs)
        
        # Ters indeks yapısını kaydetme
        save_inverted_index(inverted_index, inverted_index_path)
        print("\n\n\n Ters indeks yapısı başarıyla oluşturuldu ve kaydedildi. \n\n\n")
        
        # Ters indeks yapısını yükleme ve kontrol etme
        loaded_inverted_index = load_inverted_index(inverted_index_path)
        print("\n\n\n Ters indeks yapısından örnekler:\n\n\n ")
        word_to_display = input("Görüntülemek istediğiniz terimi girin: ")
        if word_to_display in loaded_inverted_index:
            print(f"{word_to_display}: {loaded_inverted_index[word_to_display]}")
        else:
            print("Belirtilen terim indeks içinde bulunamadı.")

        # Sorgu işleme (AND)
        query = input("\n\n\n Aramak istediğiniz terimleri girin (örnek: term1 AND term2): ")
        query_terms = query.split(' AND ')
        relevant_docs = process_query(' '.join(query_terms), loaded_inverted_index)
        
        # Sorgu işleme (OR)
        query_or = input("\n\n\n OR ile aramak istediğiniz terimleri girin (örnek: term1 OR term2): ")
        relevant_docs_or = process_or_query(query_or, loaded_inverted_index)
        display_results(relevant_docs_or, docs)
        
        # Sonuçları görüntüleme
        display_results(relevant_docs, docs)

        # TF-IDF hesaplama ve görüntüleme
        processed_documents = [doc['Processed_Words'] for doc in docs]
        docs_tfidf = calculate_tfidf_for_all_docs(processed_documents)
        print("\n\n\n Örnek TF-IDF değerleri:\n\n\n")
        print(docs_tfidf[0])

        # Değerlendirme
        queries = [
            {"ID": "001", "Words": "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft ."},
            # Buraya diğer sorguları ekleyebilirsiniz
        ]
        
        ground_truth = {
            "001": ["1066", "14", "202", "390", "746", "78", "781"]
            # Buraya diğer sorguların gerçek sonuçlarını ekleyebilirsiniz
        }

        evaluate_system(queries, ground_truth, inverted_index)

    else:
        print("Dokümanlar yüklenemedi.")

if __name__ == "__main__":
    main()
           
