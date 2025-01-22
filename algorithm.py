from abc import ABC, abstractmethod
import pandas as pd


class RankingAlgorithm(ABC):
    @abstractmethod
    def rank(self, decision_matrix: pd.DataFrame, criteria_types: dict[str, int], weights: dict[str, float]) \
            -> dict[str, tuple]:
        """
        Abstract method to rank alternatives.

        :param decision_matrix: Macierz decyzyjna (alternatywy x kryteria)
        :param criteria_types: Typy kryteriów (+1 dla maksymalizacji, -1 dla minimalizacji)
        :param weights: Wagi każdego z kryteriów (dowolne)
        :return: Słownik z rankingiem i wynikami dla alternatyw (alternatywa, (miejsce, wynik)
        """
