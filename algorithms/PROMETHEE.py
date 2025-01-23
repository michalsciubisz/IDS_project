import pandas as pd
import numpy as np
from typing import Dict
from algorithms.algorithm import RankingAlgorithm


class PROMETHEE(RankingAlgorithm):
    def __init__(self, preference_function=None):
        """
        Inicjalizacja algorytmu PROMETHEE z domyślną funkcją preferencji.
        :param preference_function: Funkcja preferencji (domyślnie różnica liniowa).
        """
        self.preference_function = preference_function or self._linear_preference

    def _linear_preference(self, d: float) -> float:
        """
        Domyślna funkcja preferencji liniowej.
        :param d: Różnica między ocenami dla kryterium.
        :return: Wartość preferencji w zakresie [0, 1].
        """
        return max(0, d)

    def linear_unicriterion_preference(self, d: float, p: float) -> float:
        """
        Funkcja preferencji liniowej - do rozważenia.
        :param d: Różnica między wartościami kryterium dla alternatyw.
        :param p: Próg, po którym preferencja osiąga wartość maksymalną.
        :return: Wartość preferencji w zakresie [0, 1].
        """
        if d <= 0:
            return 0
        elif d <= p:
            return d / p
        else:
            return 1

    def _calculate_preference_matrix(self, decision_matrix: pd.DataFrame, criteria_types: Dict[str, int]) -> np.ndarray:
        """
        Oblicza macierz preferencji dla wszystkich par alternatyw.
        :param decision_matrix: Macierz decyzyjna.
        :param criteria_types: Typy kryteriów (+1 dla maksymalizacji, -1 dla minimalizacji).
        :return: Macierz preferencji (n x n x m), gdzie n - liczba alternatyw, m - liczba kryteriów.
        """
        n, m = decision_matrix.shape
        preference_matrix = np.zeros((n, n, m))

        for i in range(n):
            for j in range(n):
                if i != j:  # Porównujemy różne alternatywy
                    for k, criterion in enumerate(decision_matrix.columns):
                        diff = (decision_matrix.iloc[i, k] - decision_matrix.iloc[j, k]) * criteria_types[criterion]
                        preference_matrix[i, j, k] = self.preference_function(diff)

        return preference_matrix

    def _calculate_flows(self, preference_matrix: np.ndarray, weights: np.ndarray) -> (
    np.ndarray, np.ndarray, np.ndarray):
        """
        Oblicza przepływy dla alternatyw.
        :param preference_matrix: Macierz preferencji.
        :param weights: Wagi kryteriów.
        :return: Przepływ wyjściowy, wejściowy i netto dla każdej alternatywy.
        """
        n = preference_matrix.shape[0]
        weighted_preferences = np.sum(preference_matrix * weights, axis=2)  # Ważone preferencje (n x n)

        outgoing_flow = np.sum(weighted_preferences, axis=1) / (n - 1)
        incoming_flow = np.sum(weighted_preferences, axis=0) / (n - 1)
        net_flow = outgoing_flow - incoming_flow

        return outgoing_flow, incoming_flow, net_flow

    def rank(self, decision_matrix: pd.DataFrame, criteria_types: Dict[str, int], weights: Dict[str, float]) -> Dict[
        str, tuple]:
        """
        Ranking alternatyw za pomocą algorytmu PROMETHEE.
        :param decision_matrix: Macierz decyzyjna (alternatywy x kryteria).
        :param criteria_types: Typy kryteriów (+1 dla maksymalizacji, -1 dla minimalizacji).
        :param weights: Wagi każdego z kryteriów.
        :return: Słownik z rankingiem i wynikami dla alternatyw (alternatywa: (miejsce, wynik)).
        """
        normalized_weights = np.array([weights[criterion] for criterion in decision_matrix.columns])
        normalized_weights /= np.sum(normalized_weights)

        preference_matrix = self._calculate_preference_matrix(decision_matrix, criteria_types)

        outgoing_flow, incoming_flow, net_flow = self._calculate_flows(preference_matrix, normalized_weights)

        alternatives = decision_matrix.index.tolist()
        results = {alt: net for alt, net in zip(alternatives, net_flow)}
        ranked_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        final_ranking = {alt: (rank + 1, score) for rank, (alt, score) in enumerate(ranked_results)}

        return final_ranking


# testing:
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

# promethee = PROMETHEE()
# ranking = promethee.rank(decision_matrix, criteria_types, weights)
# print(ranking)
