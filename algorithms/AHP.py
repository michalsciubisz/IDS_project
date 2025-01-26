import numpy as np
import pandas as pd
from algorithms.algorithm import RankingAlgorithm


class AHP(RankingAlgorithm):
    def calculate_pairwise_matrix(self, criteria_weights: list[float]) -> np.ndarray:
        """
        Generate a pairwise comparison matrix from criteria weights.

        :param criteria_weights: Weights of the criteria
        :return: Pairwise comparison matrix
        """
        n = len(criteria_weights)
        pairwise_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                pairwise_matrix[i, j] = criteria_weights[i] / criteria_weights[j]
        return pairwise_matrix

    def calculate_consistency_ratio(self, pairwise_matrix: np.ndarray) -> float:
        """
        Calculate the consistency ratio of a pairwise comparison matrix.

        :param pairwise_matrix: Pairwise comparison matrix
        :return: Consistency ratio
        """
        n = pairwise_matrix.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)
        max_eigenvalue = np.real(eigenvalues).max()
        consistency_index = (max_eigenvalue - n) / (n - 1)

        # Random Index (RI) values for different matrix sizes (1-10)
        random_index = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        ri = random_index[n - 1] if n <= 10 else 1.49  # Default RI for matrices > 10

        return consistency_index / ri if ri != 0 else 0

    def rank(self, decision_matrix: pd.DataFrame, criteria_types: dict[str, int], weights: dict[str, float]) \
            -> dict[str, tuple]:
        # Step 1: Create a pairwise comparison matrix for criteria weights
        criteria_weights = list(weights.values())
        pairwise_matrix = self.calculate_pairwise_matrix(criteria_weights)

        # Step 2: Calculate consistency ratio
        consistency_ratio = self.calculate_consistency_ratio(pairwise_matrix)
        if consistency_ratio > 0.1:
            raise ValueError(f"Consistency ratio is too high: {consistency_ratio:.2f}. Please check the pairwise comparisons.")

        # Step 3: Normalize the decision matrix based on criteria types
        normalized_matrix = decision_matrix.copy()
        for criterion, criterion_type in criteria_types.items():
            if criterion_type == 1:  # Maximization
                normalized_matrix[criterion] = decision_matrix[criterion] / decision_matrix[criterion].max()
            elif criterion_type == -1:  # Minimization
                normalized_matrix[criterion] = decision_matrix[criterion].min() / decision_matrix[criterion]

        # Step 4: Multiply the normalized matrix by weights
        weighted_matrix = normalized_matrix.copy()
        for criterion, weight in weights.items():
            weighted_matrix[criterion] *= weight

        # Step 5: Calculate the score for each alternative (sum across criteria)
        scores = weighted_matrix.sum(axis=1)

        # Step 6: Rank the alternatives based on scores
        rankings = scores.rank(ascending=False).astype(int)

        # Prepare the result as a dictionary
        result = {
            alternative: (rank, score)
            for alternative, rank, score in zip(weighted_matrix.index, rankings, scores)
        }

        return result


# testing:
decision_matrix = pd.DataFrame({
    "Accuracy": [0.95, 0.91, 0.93],
    "F1 Score": [0.91, 0.99, 0.91],
    "Time": [0.90, 0.1, 0.25]
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

# ahp = AHP()
# ranking = ahp.rank(decision_matrix, criteria_types, weights)
# print(ranking)
