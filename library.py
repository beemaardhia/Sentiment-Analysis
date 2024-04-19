import pandas as pd
import numpy as np

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN :
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x) :
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        label_counts = {}
        for label in k_nearest_labels :
            if label in label_counts :
                label_counts[label] =+ 1
            else :
                label_counts[label] = 1

        most_common_labels = max(label_counts, key=label_counts.get)
        return most_common_labels
    
class TFIDF :
    def __init__(self):
        self.vocab_ = {}
        self.idf_ = []

    def fit(self, documents) :
        df = {}
        for document in documents :
            words = set(document.split())
            for word in words :
                df[word] = df.get(word, 0) + 1
        
        self.vocab_ = {word: idx for idx, word in enumerate(sorted(df.keys()))}
        total_documents = len(documents)
        self.idf_ = [np.log((total_documents + 1) / (df[word] + 1)) + 1 for word in sorted(df.keys())]

    def transform(self, documents) :
        rows = len(documents)
        cols = len(self.vocab_)
        tfidf_matrix = np.zeros((rows, cols))

        for row, document in enumerate(documents) :
            word_counts = {}
            words = document.split()
            for word in words :
                if word in self.vocab_ :
                    idx = self.vocab_[word]
                    word_counts[idx] = word_counts.get(word, 0) + 1
            
            for idx, count in word_counts.items() :
                tf = count / len(words)
                tfidf_matrix[row, idx] = tf * self.idf_[idx]

        return tfidf_matrix
    
    def get_names_features_out(self) :
        return list(self.vocab_.keys())
