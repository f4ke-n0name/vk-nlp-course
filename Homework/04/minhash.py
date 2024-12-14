import re
import pandas as pd
import numpy as np


class MinHash:
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold

    def preprocess_text(self, text: str) -> str:
        return re.sub("( )+|(\n)+"," ",text).lower()

    def tokenize(self, text: str) -> set:
        text = self.preprocess_text(text)      
        return set(text.split(' '))
    
    def get_occurrence_matrix(self, corpus_of_texts: list[set]) -> pd.DataFrame:
        '''
        Получение матрицы вхождения токенов. Строки - это токены, столбы это id документов.
        id документа - нумерация в списке начиная с нуля
        '''
        processed_texts = [self.tokenize(self.preprocess_text(doc)) for doc in corpus_of_texts]
        unique_words = sorted(set(word for doc in processed_texts for word in doc))
        data = []
        for word in unique_words:
            row_count = [(1 if word in doc else None) for doc in processed_texts]
            data.append(row_count)
        df = pd.DataFrame(data, columns=list(range(len(corpus_of_texts))))

        df.sort_index(inplace=True)
        return df

    
    def is_prime(self, a):
        if a % 2 == 0:
            return a == 2
        d = 3
        while d * d <= a and a % d != 0:
            d += 2
        return d * d > a
    
    def get_new_index(self, x: int, permutation_index: int, prime_num_rows: int) -> int:
        '''
        Получение перемешанного индекса.
        values_dict - нужен для совпадения результатов теста, а в общем случае используется рандом
        prime_num_rows - здесь важно, чтобы число было >= rows_number и было ближайшим простым числом к rows_number

        '''
        values_dict = {
            'a': [3, 4, 5, 7, 8],
            'b': [3, 4, 5, 7, 8] 
        }
        a = values_dict['a'][permutation_index]
        b = values_dict['b'][permutation_index]
        return (a*(x+1) + b) % prime_num_rows 
    
    
    def get_minhash_similarity(self, array_a: np.array, array_b: np.array) -> float:
        '''
        Вовзращает сравнение minhash для НОВЫХ индексов. То есть: приходит 2 массива minhash:
            array_a = [1, 2, 1, 5, 3]
            array_b = [1, 3, 1, 4, 3]

            на выходе ожидаем количество совпадений/длину массива, для примера здесь:
            у нас 3 совпадения (1,1,3), ответ будет 3/5 = 0.6
        '''
        lenght = array_a.size
        similarity = 0.0
        for i in range(array_a.size):
            if (array_a[i] == array_b[i]):
                similarity += 1
        return similarity / lenght

    
    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        result = []
        matrix = self.get_similar_matrix(min_hash_matrix)
        for i in range(len(min_hash_matrix[0])):
            for j in range(i + 1, len(min_hash_matrix[0])):
                if matrix[i][j] > self.threshold:
                    result.append((i, j))
        return result
    
    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''
        matrix = []
        min_hash_matrix_T = min_hash_matrix.T
        for i in range(len(min_hash_matrix_T)):
            column = []
            for j in range(len(min_hash_matrix_T)):
                intersect = self.get_minhash_similarity(min_hash_matrix_T[i], min_hash_matrix_T[j])
                column.append(intersect)
            matrix.append(column)       
        return matrix
     
    
    def get_minhash(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей. MinHash содержит в себе новые индексы. 

        new index = (2*(index +1) + 3) % 3 
        
        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу 
        [1, 0, 1]
        [1, 0, 1]
        [0, 1, 1]

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 0
        Doc2 : 2
        Doc3 : 0
        '''
        prime_num_rows = len(occurrence_matrix)
        while not self.is_prime(prime_num_rows):
            prime_num_rows += 1
        result = []
        count_documents = len(occurrence_matrix.columns)
        for permutation_index in range(self.num_permutations):
            keys = [self.get_new_index(i, permutation_index, prime_num_rows) for i in range(len(occurrence_matrix))]
            shuffled_df = occurrence_matrix.copy()
            shuffled_df['new_index'] = keys
            shuffled_df = shuffled_df.sort_values(by='new_index')
            shuffled_df = shuffled_df.drop(columns=['new_index'])
            row = [shuffled_df[i].first_valid_index() for i in range(count_documents)]
            result.append(row)
        return np.array(result)

    
    def run_minhash(self,  corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs

class MinHashJaccard(MinHash):
    def __init__(self, num_permutations: int, threshold: float):
        self.num_permutations = num_permutations
        self.threshold = threshold
    
    
    def get_jaccard_similarity(self, set_a: set, set_b: set) -> float:
        '''
        Вовзращает расстояние Жаккарда для двух сетов. 
        '''
        if len(set_a) == 0 and len(set_b) == 0:
            return 0.0
        intersection_of_sets = set_a.intersection(set_b)
        union_of_sets = set_a.union(set_b)
        return 1 - float(len(intersection_of_sets)) / len(union_of_sets)

    
    def get_similar_pairs(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает список из таплов индексов похожих документов, похожесть которых > threshold.
        '''
        return super().get_similar_pairs(min_hash_matrix) 
    
    def get_similar_matrix(self, min_hash_matrix) -> list[tuple]:
        '''
        Находит похожих кандидатов. Отдает матрицу расстояний
        '''
                
        return super().get_similar_matrix(min_hash_matrix)
     
    
    def get_minhash_jaccard(self, occurrence_matrix: pd.DataFrame) -> np.array:
        '''
        Считает и возвращает матрицу мин хешей. Но в качестве мин хеша выписываем минимальный исходный индекс, не новый.
        В такой ситуации можно будет пользоваться расстояние Жаккрада.

        new index = (2*(index +1) + 3) % 3 
        
        Пример для 1 перемешивания:
        [0, 1, 1] new index: 2
        [1, 0, 1] new index: 1
        [1, 0, 1] new index: 0

        отсортируем по новому индексу 
        [1, 0, 1] index: 2
        [1, 0, 1] index: 1
        [0, 1, 1] index: 0

        Тогда первый элемент minhash для каждого документа будет:
        Doc1 : 2
        Doc2 : 0
        Doc3 : 2
        
        '''
        return super().get_minhash(occurrence_matrix)

    
    def run_minhash(self,  corpus_of_texts: list[str]):
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash_jaccard(occurrence_matrix)
        similar_pairs = self.get_similar_pairs(minhash)
        similar_matrix = self.get_similar_matrix(minhash)
        print(similar_matrix)
        return similar_pairs

    
    
