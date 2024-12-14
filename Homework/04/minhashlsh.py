import pandas as pd
import numpy as np


from minhash import MinHash

class MinHashLSH(MinHash):
    def __init__(self, num_permutations: int, num_buckets: int, threshold: float):
        self.num_permutations = num_permutations
        self.num_buckets = num_buckets
        self.threshold = threshold
        
    def get_buckets(self, minhash: np.array) -> np.array:
        '''
        Возвращает массив из бакетов, где каждый бакет представляет собой N строк матрицы сигнатур.
        '''
        step = len(minhash) // self.num_buckets
        result = []
        left = 0

        for i in range(self.num_buckets):
            cur_len = step + (1 if (len(minhash) - left) % (self.num_buckets - i) > 0 else 0)
            result.append(minhash[left:left + cur_len])
            left += cur_len
            if left >= len(minhash):
                break

        return result
    
    def get_similar_candidates(self, buckets) -> list[tuple]:
        '''
        Находит потенциально похожих кандижатов.
        Кандидаты похожи, если полностью совпадают мин хеши хотя бы в одном из бакетов.
        Возвращает список из таплов индексов похожих документов.
        '''
        similar_candidates = set()
        for bucket in buckets:
            value_to_indices = {}
            for idx in range(bucket.shape[1]):
                value_tuple = tuple(bucket[:, idx])
                if value_tuple not in value_to_indices:
                    value_to_indices[value_tuple] = []
                value_to_indices[value_tuple].append(idx)

            for indices in value_to_indices.values():
                if len(indices) > 1:
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            similar_candidates.add((indices[i], indices[j]))
        return similar_candidates
        
    def run_minhash_lsh(self, corpus_of_texts: list[str]) -> list[tuple]:
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        buckets = self.get_buckets(minhash)
        similar_candidates = self.get_similar_candidates(buckets)
        
        return set(similar_candidates)
    
    
