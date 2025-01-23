from algorithm import RankingAlgorithm
import pandas as pd
import numpy as np

class EntropyMethod(RankingAlgorithm):
    def rank(self, decision_matrix: pd.DataFrame, criteria_types: dict[str, int], weights: dict[str, float]) -> dict[str, tuple]:
        """
        Implementacja rankingu dla algorytmu Entropy.
        :param decision_matrix: Macierz decyzyjna (alternatywy x kryteria).
        :param criteria_types: Typy kryteriów (ignorowane w Entropy, ale zachowane dla spójności interfejsu).
        :param weights: Wagi kryteriów (zdefiniowane przez użytkownika lub wyznaczone z macierzy porównań parowych).
        :return: Słownik z rankingiem i wynikami dla alternatyw (alternatywa, (miejsce, wynik)).
        """
        normalized_matrix = decision_matrix / decision_matrix.sum(axis=0)
        
        n = len(decision_matrix) 
        entropy_values = []

        for col in normalized_matrix.columns:
            p_ij = normalized_matrix[col]  
            entropy = -np.sum(p_ij * np.log(p_ij + 1e-10)) / np.log(n) 
            entropy_values.append(entropy)

        entropy_values = np.array(entropy_values)
        weights = (1 - entropy_values) / np.sum(1 - entropy_values)
        

        weighted_matrix = normalized_matrix * weights
        scores = weighted_matrix.sum(axis=1)

        ranking = pd.Series(scores, index=decision_matrix.index).sort_values(ascending=False)
        return {name: (rank + 1, score) for rank, (name, score) in enumerate(ranking.items())}

# Example usage:
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

entropy_method = EntropyMethod()
ranking = entropy_method.rank(decision_matrix, criteria_types, weights)

print(ranking)