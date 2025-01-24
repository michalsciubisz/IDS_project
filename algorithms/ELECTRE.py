import pandas as pd
import numpy as np
from algorithms.algorithm import RankingAlgorithm

class ELECTRE(RankingAlgorithm):
    def __init__(self):
        super().__init__()
        
    def _normalize(self, decision_matrix: pd.DataFrame, criteria_types: dict[str, int]) -> pd.DataFrame:
        """
        Normalizacja macierzy alternatyw, na bazie kryteriów (tego czy są maksymalizowane czy minimalizowane).
        :param decision_matrix: Macierz decyzyjna (alternatywy x kryteria).
        :param criteria_types: Typy kryteriów (ignorowane w AHP, ale zachowane dla spójności interfejsu).
        :reutrn: Zbiór znormalizowanych danych.
        """
        normalized_matrix = decision_matrix.copy()
        for criterion, criterion_type in criteria_types.items():
            if criterion_type == 1:  # Maximization
                normalized_matrix[criterion] = decision_matrix[criterion] / decision_matrix[criterion].max()
            elif criterion_type == -1:  # Minimization
                normalized_matrix[criterion] = decision_matrix[criterion].min() / decision_matrix[criterion]
        return normalized_matrix

    def _calculate_concordance_discordance(self, normalized_matrix: pd.DataFrame,
                                           weights: dict[str, float]) -> tuple:
        """
        Obliczenie zgodności oraz niezgnodności macierzy.
        :param normalized_matrix: Macierz decyzyjna znormalizowana (alternatywy x kryteria).
        :param weights: Wagi kryteriów (zdefiniowane przez użytkownika lub wyznaczone z macierzy porównań parowych).
        :return: Dwie macierzy zgodności oraz niezgodności (alternatwywy x alternatywy).
        """
        num_alternatives = normalized_matrix.shape[0]
        concordance_matrix = np.zeros((num_alternatives, num_alternatives))
        discordance_matrix = np.zeros((num_alternatives, num_alternatives))

        for i in range(num_alternatives):
            for j in range(num_alternatives):
                if i != j:
                    concordance = 0
                    discordance = 0
                    for criterion in normalized_matrix.columns:
                        if normalized_matrix.iloc[i][criterion] > normalized_matrix.iloc[j][criterion]:
                            concordance += weights[criterion]
                        elif normalized_matrix.iloc[i][criterion] < normalized_matrix.iloc[j][criterion]:
                            discordance += weights[criterion]
                    concordance_matrix[i, j] = concordance
                    discordance_matrix[i, j] = discordance

        return concordance_matrix, discordance_matrix

    def _calculate_electre_scores(self, concordance_matrix: np.ndarray, discordance_matrix: np.ndarray) -> dict:
        """
        Obliczenie wartości wskaźnika Electre.
        :param concordance_matrix: Macierz zgodności (alternatwywy x alternatywy).
        :param dicsordance: Macierz niezgodności (alternatwywy x alternatywy).
        :return: Słownik wartości wskaźnika Electre dla poszczegónych wierszy {indeks, wartość wskaźnika}.
        """
        electre_scores = {}
        for i in range(concordance_matrix.shape[0]):
            score = 0
            for j in range(concordance_matrix.shape[1]):
                if concordance_matrix[i, j] > discordance_matrix[i, j]:
                    score += 1
            electre_scores[i] = score
        return electre_scores

    def _rank_alternatives(self, electre_scores: dict, index: pd.Index) -> dict:
        """
        Ocena alternatyw ze względu na wartość wskaźnika.
        :param electre_scores: Słownik wartości wskaźnika Electre dla poszczegónych wierszy {indeks, wartość wskaźnika}.
        :param index: Index of the original decision matrix.
        :return: Słownik z rankingiem alternatyw {alternatywa, (miejsce, wynik)}.
        """
        sorted_scores = sorted(electre_scores.items(), key=lambda item: item[1], reverse=True)
        ranked_alternatives = {
            index[i]: (rank + 1, score)  # Use the provided index
            for rank, (i, score) in enumerate(sorted_scores)
        }
        return ranked_alternatives
    
    def rank(self, decision_matrix: pd.DataFrame, criteria_types: dict[str, int], weights: dict[str, float]) \
        -> dict[str, tuple]:
        """
        Implementacja rankingu dla algorytmu ELECTRE.
        :param decision_matrix: Macierz decyzyjna (alternatywy x kryteria).
        :param criteria_types: Typy kryteriów.
        :param weights: Wagi kryteriów (zdefiniowane przez użytkownika).
        :return: Słownik z rankingiem i wynikami dla alternatyw (alternatywa, (miejsce, wynik)).
        """
        normalized_matrix = self._normalize(decision_matrix, criteria_types)
        concordance_matrix, discordance_matrix = self._calculate_concordance_discordance(
            normalized_matrix, weights
        )
        electre_scores = self._calculate_electre_scores(concordance_matrix, discordance_matrix)
        ranking = self._rank_alternatives(electre_scores, decision_matrix.index)  # Pass the index here
        return ranking


# Example usage
decision_matrix = pd.DataFrame({
    "Accuracy": [0.95, 0.91, 0.93],
    "F1 Score": [0.89, 0.89, 0.91],
    "Time": [0.12, 0.34, 0.25]
}, index=["Logistic Regression", "Random Forest", "XGBoost"])

criteria_types = {
    "Accuracy": 1,
    "F1 Score": 1,
    "Time": -1
}

weights = {
    "Accuracy": 0.4,
    "F1 Score": 0.4,
    "Time": 0.2
}

# electre = Electre()
# ranking = electre.rank(decision_matrix, criteria_types, weights)
# print(ranking)