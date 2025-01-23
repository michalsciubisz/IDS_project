import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file):
    df = pd.read_csv(file)
    return df

def plot_heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.drop(columns='Model name').corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

def ahp_method(df):
    st.write("AHP method would go here")
    return df 

def vikor_method(df):
    st.write("VIKOR method would go here")
    return df

# Main Streamlit app
def main():
    st.title("Ocena jakości działania klasyfikatorów przy użyciu metod MCDM")
    
    # Upload CSV
    uploaded_file = st.file_uploader("Wybierz plik CSV", type="csv")

    with st.expander("Train and Evaluate Models"):
        pass
    
    if uploaded_file is not None:
        # Load and display the data
        df = load_data(uploaded_file)
        st.subheader("Classifier Evaluation Results")
        st.dataframe(df)
        
        # Plot heatmap of correlations
        st.subheader("Heatmap of Criteria Correlations")
        plot_heatmap(df)
        
        # Dropdown to select MCDM method
        method = st.selectbox("Select MCDM Method", ["AHP", "VIKOR", "PROMETHEE", "ELECTRE", "Entropy"])
        
        # Apply the selected method
        if method == "AHP":
            result = ahp_method(df)
        elif method == "VIKOR":
            result = vikor_method(df)
        # Add other methods here
        
        # Display the result after evaluation
        st.subheader(f"Results after applying {method} method")
        st.write(result)

# Run the app
if __name__ == "__main__":
    main()