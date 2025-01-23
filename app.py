from classifier import *

from algorithms.AHP import *
from algorithms.VIKOR import *
from algorithms.PROMETHEE import *
from algorithms.ELECTRE import *
from algorithms.ENTROPY import *

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _load_data(file):
    df = pd.read_csv(file)
    return df

def _ahp_method(df):
    #TODO implement
    return df 

def _vikor_method(df):
    #TODO implement
    return df

def _promethee_method(df):
    #TODO implement
    return df

def _electre_method(df):
    #TODO implement
    return df

def _entropy_method(df):
    #TODO implement
    return df

#tutaj dodaje się wszystko
def main():
    st.title("Ocena jakości działania klasyfikatorów przy użyciu metod MCDM")

    #wytrenowanie zbioru na bieżąco i użycie go do dalszej pracy
    with st.expander("Wytrenuj klasyfikatory i wykorzystaj je dalej!"):
        st.write("Tutaj wybierz plik do wytrenowania oraz ocenić działanie algorytmów.")

        training_file = st.file_uploader("Wybierz plik CSV z folderu data (test_prepared lub train_prepared)", type='csv')

        num_iteration = st.slider("Wybierz liczbę iteracji (~160 sekund na iterację)", min_value=1, max_value=5)

        if st.button("Rozpocznij trenowanie"):
            if training_file is not None:
                training_df = _load_data(training_file)

                classifiers = ["logistic", "decision_tree", "random_forest", "knn", "naive_bayes", 
                   "gradient_boost", "mlp", "xgbc", "lgbmc"]
    
                models = [ClassifierFactory.create_classifier(name, training_df) for name in classifiers]

                for _ in range(num_iteration):
                    for model in models:
                        model._prep_split_data()
                        model._train_and_save()
        
        if st.button("Wyczyść zawartość pliku!"):
            header = "Model name,Accuracy,Sensitivity,Precision,F1 Score,Specificity,Balanced accuracy,MCC,False Positive Rate,False Negative Rate,Time"
            file_path = "evaluation_results.csv"
            if os.path.exists(file_path):
                # Clear the file and write the header
                with open(file_path, 'w', newline='', encoding='utf-8') as file:
                    file.write(header + '\n')  # Write the header row
                st.success("Plik został wyczyszczony i nagłówek został dodany!")
            else:
                # If file does not exist, create it and write the header
                with open(file_path, 'w', newline='', encoding='utf-8') as file:
                    file.write(header + '\n')  # Write the header row
                st.warning("Plik evaluation_results.csv nie istniał, więc został utworzony z nagłówkiem!")
    
    classify_results_file = st.file_uploader("Wybierz plik CSV", type="csv")

    if classify_results_file is not None:
        #wczytywanie danych
        df = _load_data(classify_results_file)
        st.subheader("Classifier Evaluation Results")
        st.dataframe(df)
        
        # Dropdown to select MCDM method
        # method = st.selectbox("Select MCDM Method", ["AHP", "VIKOR", "PROMETHEE", "ELECTRE", "Entropy"])
        
        # Apply the selected method
        # if method == "AHP":
        #     result = _ahp_method(df)
        # elif method == "VIKOR":
        #     result = _vikor_method(df)
        # Add other methods here
        
        # Display the result after evaluation
        # st.subheader(f"Results after applying {method} method")
        # st.write(result)

# Run the app
if __name__ == "__main__":
    main()