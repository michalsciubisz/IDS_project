from classifier import *

from algorithms.AHP import *
from algorithms.VIKOR import *
from algorithms.PROMETHEE import *
from algorithms.ELECTRE import *
from algorithms.ENTROPY import *

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

if __name__ == "__main__":
    main()