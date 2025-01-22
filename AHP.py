import pandas as pd
from algorithm import RankingAlgorithm

class AHP(RankingAlgorithm):
    def __init__(self):
        pass

    def rank(self, decision_matrix: pd.DataFrame, criteria_types: dict[str, int], weights: dict[str, float]) \
            -> dict[str, tuple]:
        """
        Implementacja rankingu dla algorytmu AHP.
        :param decision_matrix: Macierz decyzyjna (alternatywy x kryteria).
        :param criteria_types: Typy kryteriów (ignorowane w AHP, ale zachowane dla spójności interfejsu).
        :param weights: Wagi kryteriów (zdefiniowane przez użytkownika lub wyznaczone z macierzy porównań parowych).
        :return: Słownik z rankingiem i wynikami dla alternatyw (alternatywa, (miejsce, wynik)).
        """

        # Normalizacja macierzy decyzyjnej dla każdego kryterium
        normalized_matrix = decision_matrix.copy()
        for criterion in decision_matrix.columns:
            if criteria_types[criterion] == 1:  # Maksymalizacja
                normalized_matrix[criterion] = decision_matrix[criterion] / decision_matrix[criterion].sum()
            elif criteria_types[criterion] == -1:  # Minimalizacja
                normalized_matrix[criterion] = decision_matrix[criterion].min() / decision_matrix[criterion]

        # Mnożenie przez wagi kryteriów
        weighted_matrix = normalized_matrix.copy()
        for criterion in weights.keys():
            weighted_matrix[criterion] *= weights[criterion]

        # Sumowanie ważonych wyników dla każdej alternatywy
        final_scores = weighted_matrix.sum(axis=1)

        # Tworzenie rankingu
        alternatives = decision_matrix.index.tolist()
        results = pd.DataFrame({'Alternative': alternatives, 'Score': final_scores})
        results = results.sort_values(by='Score', ascending=False).reset_index(drop=True)
        rankings = {row['Alternative']: (rank + 1, row['Score']) for rank, row in results.iterrows()}

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

ahp = AHP()
ranking = ahp.rank(decision_matrix, criteria_types, weights)
print(ranking)
