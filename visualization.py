import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_alternatives(df):
    st.subheader("Wizualizacje alternatyw względem kryteriów")

    # Kryteria maksymalizacyjne i minimalizacyjne
    maximize_criteria = [
        "Accuracy", "Sensitivity", "Precision", "F1 Score",
        "Specificity", "Balanced accuracy", "MCC"
    ]
    minimize_criteria = ["False Positive Rate", "False Negative Rate", "Time"]

    # Przygotowanie unikalnych kolorów dla każdej alternatywy
    unique_models = df["Model name"].unique()
    colors = px.colors.qualitative.Set2  # Wybór palety kolorów
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(unique_models)}

    # Wykres 3D dla maksymalizacyjnych (z możliwością wyboru kryteriów)
    st.write("### Wykres 3D: Kryteria maksymalizacyjne")
    default_max_criteria = maximize_criteria[:3]  # Domyślny wybór 3 kryteriów
    max_x = st.selectbox("Kryterium na osi X", maximize_criteria, index=maximize_criteria.index(default_max_criteria[0]))
    max_y = st.selectbox("Kryterium na osi Y", maximize_criteria, index=maximize_criteria.index(default_max_criteria[1]))
    max_z = st.selectbox("Kryterium na osi Z", maximize_criteria, index=maximize_criteria.index(default_max_criteria[2]))

    fig1 = px.scatter_3d(
        df, x=max_x, y=max_y, z=max_z, color="Model name",
        title=f"Wizualizacja 3D dla kryteriów maksymalizacyjnych ({max_x}, {max_y}, {max_z})",
        labels={"x": max_x, "y": max_y, "z": max_z},
        color_discrete_map=model_colors  # Przypisanie kolorów
    )
    fig1.update_traces(marker=dict(size=8))
    st.plotly_chart(fig1, use_container_width=True)

    # Wykres 3D dla minimalizacyjnych (bez możliwości wyboru kryteriów)
    st.write("### Wykres 3D: Kryteria minimalizacyjne")
    min_x, min_y, min_z = minimize_criteria  # Kryteria na sztywno
    fig2 = px.scatter_3d(
        df, x=min_x, y=min_y, z=min_z, color="Model name",
        title=f"Wizualizacja 3D dla kryteriów minimalizacyjnych ({min_x}, {min_y}, {min_z})",
        labels={"x": min_x, "y": min_y, "z": min_z},
        color_discrete_map=model_colors  # Przypisanie kolorów
    )
    fig2.update_traces(marker=dict(size=8))
    st.plotly_chart(fig2, use_container_width=True)

    # Sumowanie wartości maksymalizacyjnych i minimalizacyjnych dla wykresów słupkowych
    st.write("### Sumaryczna ocena alternatyw dla maksymalizacyjnych i minimalizacyjnych kryteriów")
    max_sum = df[maximize_criteria].sum(axis=1).tolist()  # Suma maksymalizacyjnych
    min_sum = df[minimize_criteria].sum(axis=1).tolist()  # Suma minimalizacyjnych

    # Tworzenie list alternatyw i odpowiadających im wartości
    models = df["Model name"].tolist()
    max_data = list(zip(models, max_sum))
    min_data = list(zip(models, min_sum))

    # Sortowanie listy wyników
    max_sorted = sorted(max_data, key=lambda x: x[1], reverse=True)  # Sortowanie malejąco
    min_sorted = sorted(min_data, key=lambda x: x[1])  # Sortowanie rosnąco

    # Wykres słupkowy dla maksymalizacyjnych
    st.write("#### Suma wartości dla kryteriów maksymalizacyjnych")
    fig3 = px.bar(
        pd.DataFrame(max_sorted, columns=["Model name", "Max_Sum"]),
        x="Model name", y="Max_Sum", color="Model name",
        title="Suma maksymalizacyjnych kryteriów (posortowane)",
        labels={"Max_Sum": "Suma maksymalizacyjnych", "Model name": "Model"},
        color_discrete_map=model_colors  # Przypisanie kolorów
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Wykres słupkowy dla minimalizacyjnych
    st.write("#### Suma wartości dla kryteriów minimalizacyjnych")
    fig4 = px.bar(
        pd.DataFrame(min_sorted, columns=["Model name", "Min_Sum"]),
        x="Model name", y="Min_Sum", color="Model name",
        title="Suma minimalizacyjnych kryteriów (posortowane)",
        labels={"Min_Sum": "Suma minimalizacyjnych", "Model name": "Model"},
        color_discrete_map=model_colors  # Przypisanie kolorów
    )
    st.plotly_chart(fig4, use_container_width=True)



def plot_mcdm_results(method, rankings):
    # Convert rankings into a DataFrame
    method_data = [(classifier, rank, score) for classifier, (rank, score) in rankings.items()]
    method_df = pd.DataFrame(method_data, columns=["Classifier", "Rank", "Score"])
    
    # Sort by rank to ensure proper order
    method_df = method_df.sort_values(by="Rank")
    
    # Create a horizontal bar plot for each method's results
    fig = px.bar(
        method_df, 
        x="Score", 
        y="Classifier", 
        color="Classifier", 
        orientation="h",  # Horizontal bars
        title=f"Classifier Scores for {method}",
        labels={"Score": "Score", "Classifier": "Classifier"},
        color_discrete_sequence=px.colors.qualitative.Set2  # Use the Set2 color palette
    )
    
    # Update the layout for better readability
    fig.update_layout(
        xaxis_title="Score",
        yaxis_title="Classifier",
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=100, r=100, t=50, b=50)
    )
    
    # Display the plot within the Streamlit app
    st.plotly_chart(fig, use_container_width=True)