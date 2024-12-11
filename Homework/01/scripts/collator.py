import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

class Collator:
    """
    Класс Collator используется для дополнения (padding) списков разной длины
    до одинаковой длины с использованием заданного значения padding_value.

    Аргументы:
        padding_value (int): Значение, которое будет использоваться для дополнения
        (padding) последовательностей до одинаковой длины.
    """

    def __init__(self, padding_value: int):
        self.padding_value = padding_value
        """
        Инициализирует Collator с заданным значением для дополнения.

        Аргументы:
            padding_value (int): Значение для padding.
        """

    def __call__(self, data: List[List[int]]) -> Tensor:
        data = [torch.tensor(x, dtype=torch.long) for x in data]
        data = pad_sequence(data, batch_first=True , padding_value=self.padding_value)
        return data
