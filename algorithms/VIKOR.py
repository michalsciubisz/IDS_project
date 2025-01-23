import pandas as pd
import numpy as np
from algorithms.algorithm import RankingAlgorithm


class VIKOR(RankingAlgorithm):
    def __init__(self, v: float = 0.5):
        """
        Inicjalizacja algorytmu VIKOR.
        :param v: Parametr równowagi kompromisu (domyślnie 0.5).
        """
        self.v = v

    def rank(self, decision_matrix: pd.DataFrame, criteria_types: dict[str, int], weights: dict[str, float]) \
            -> dict[str, tuple]:
        """
        Implementacja rankingu dla algorytmu VIKOR.
        :param decision_matrix: Macierz decyzyjna (alternatywy x kryteria).
        :param criteria_types: Typy kryteriów (+1 dla maksymalizacji, -1 dla minimalizacji).
        :param weights: Wagi każdego z kryteriów.
        :return: Słownik z rankingiem i wynikami dla alternatyw (alternatywa, (miejsce, wynik)).
        """
        best = {}
        worst = {}
        for criterion, crit_type in criteria_types.items():
            if crit_type == 1:
                best[criterion] = decision_matrix[criterion].max()
                worst[criterion] = decision_matrix[criterion].min()
            elif crit_type == -1:
                best[criterion] = decision_matrix[criterion].min()
                worst[criterion] = decision_matrix[criterion].max()

        S = []
        R = []
        for i, row in decision_matrix.iterrows():
            s_i = 0
            r_i = -np.inf
            for criterion, crit_type in criteria_types.items():
                normalized_diff = (best[criterion] - row[criterion]) / (best[criterion] - worst[criterion])
                weighted_diff = weights[criterion] * normalized_diff
                s_i += weighted_diff
                r_i = max(r_i, weighted_diff)
            S.append(s_i)
            R.append(r_i)

        S_star = min(S)
        S_minus = max(S)
        R_star = min(R)
        R_minus = max(R)

        Q = []
        for s_i, r_i in zip(S, R):
            q_i = (self.v * (s_i - S_star) / (S_minus - S_star)) + \
                  ((1 - self.v) * (r_i - R_star) / (R_minus - R_star))
            Q.append(q_i)

        alternatives = decision_matrix.index.tolist()
        results = pd.DataFrame({'Alternative': alternatives, 'S': S, 'R': R, 'Q': Q})
        results = results.sort_values(by='Q').reset_index(drop=True)
        rankings = {row['Alternative']: (rank + 1, row['Q']) for rank, row in results.iterrows()}

        return rankings


# testing:
decision_matrix = pd.DataFrame({
    "Accuracy": [0.95, 0.91, 0.93],
    "F1 Score": [0.91, 0.89, 0.91],
    "Time": [0.90, 0.34, 0.25]
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

# vikor = VIKOR()
# ranking = vikor.rank(decision_matrix, criteria_types, weights)
# print(ranking)
