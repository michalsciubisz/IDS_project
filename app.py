from classifier import *

from algorithms.AHP import *
from algorithms.VIKOR import *
from algorithms.PROMETHEE import *
from algorithms.ELECTRE import *
from algorithms.ENTROPY import *
from visualization import visualize_alternatives, plot_mcdm_results

from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import pandas as pd

#caching zeby troche przyspieszyc dzialanie
@st.cache_data
def _load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def _train_classifiers(training_df, num_iteration):
    classifiers = ["logistic", "decision_tree", "random_forest", "knn", "naive_bayes", 
                   "gradient_boost", "mlp", "xgbc", "lgbmc"]
    models = [ClassifierFactory.create_classifier(name, training_df) for name in classifiers]

    def _train_model(model):
        model._prep_split_data()
        model._train_and_save()

    with ThreadPoolExecutor() as executor:
        for _ in range(num_iteration):
            executor.map(_train_model, models)

    return models

def _adjust_weights(headers):
    """Dynamic weight adjustment UI for criteria."""
    weights = {}
    with st.expander("Adjust Weights for Criteria"):
        for criterion in headers:
            weights[criterion] = st.slider(
                f"Set weight for {criterion}",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )
    return weights

def _execute_mcdm_algorithms(matrix, criteria_types, weights):
    """Run all MCDM algorithms and return rankings."""
    algorithms = {
        "AHP": AHP(),
        "VIKOR": VIKOR(),
        "PROMETHEE": PROMETHEE(),
        "ELECTRE": ELECTRE(),
        "ENTROPY": ENTROPY(),
    }
    
    results = {}
    for name, algorithm in algorithms.items():
    
        results[name] = algorithm.rank(matrix, criteria_types, weights)
    
    return results

#tutaj cala glowna logika aplikacji
def main():
    st.title("Ocena jakości działania klasyfikatorów przy użyciu metod MCDM")

    #czesc odpowiedzialna za trening
    with st.expander("Wytrenuj klasyfikatory i wykorzystaj je dalej!"):
        training_file = st.file_uploader("Wybierz plik CSV z folderu data", type='csv')
        num_iteration = st.slider("Wybierz liczbę iteracji", min_value=1, max_value=5)

        if st.button("Rozpocznij trenowanie"):
            if training_file:
                training_df = _load_data(training_file)
                with st.spinner("Trenowanie klasyfikatorów..."):
                    _train_classifiers(training_df, num_iteration)
                st.success("Trenowanie zakończone!")

        if st.button("Wyczyść zawartość pliku!"):
            header = "Model name,Accuracy,Sensitivity,Precision,F1 Score,Specificity,Balanced accuracy,MCC,False Positive Rate,False Negative Rate,Time"
            file_path = "evaluation_results.csv"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(header + '\n')
            st.success("Plik został wyczyszczony!")

    #czesc z algorytmami
    classify_results_file = st.file_uploader("Wybierz plik CSV do oceny", type="csv")
    if classify_results_file:
        df = _load_data(classify_results_file)
        st.subheader("Wybór najlepszego klasyfikatora")
        st.dataframe(df)
        visualize_alternatives(df)

         # Adjust weights
        headers = list(df.columns)[1:]  # Skip the first column (e.g., Model name)
        weights = _adjust_weights(headers)

        # Determine criteria types
        criteria_types = {
            "Accuracy": 1,
            "Sensitivity": 1,
            "Precision": 1,
            "F1 Score": 1,
            "Specificity": 1,
            "Balanced accuracy": 1,
            "MCC": 1,
            "False Positive Rate": -1,
            "False Negative Rate": -1,
            "Time": -1,
        }

        # Execute MCDM algorithms
        if st.button("Evaluate with MCDM Algorithms"):
            alternatives = (
                df.loc[:, ~df.columns.isin(["Max_Sum", "Min_Sum"])]
                .set_index("Model name")
                .copy()
            )
            
            with st.spinner("Evaluating classifiers..."):
                results = _execute_mcdm_algorithms(alternatives, criteria_types, weights)

            st.subheader("MCDM Rankings")
            
            for method, rankings in results.items():
                comparison_data = []
                for classifier, (rank, score) in rankings.items():
                    comparison_data.append({
                        "Classifier": classifier,
                        "Rank": rank,
                        "Score": score
                    })
                comparison_df = pd.DataFrame(comparison_data)
                st.write(f"### {method} Result:")
                st.dataframe(comparison_df)

                plot_mcdm_results(method, rankings)

if __name__ == "__main__":
    main()