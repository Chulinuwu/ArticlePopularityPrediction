import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_prediction_files(results_folder):
    """
    Load prediction CSV files from results folder
    
    Parameters:
    - results_folder: Path to folder containing prediction files
    
    Returns:
    - Dictionary of DataFrames with predictions
    """
    predictions = {}
    for filename in os.listdir(results_folder):
        if filename.endswith('.csv'):
            model_name = filename.replace('predicted_vs_actual_', '').replace('.csv', '')
            file_path = os.path.join(results_folder, filename)
            predictions[model_name] = pd.read_csv(file_path)
    return predictions

def create_comparison_plot(df, model_name):
    """
    Create comparison plot using Seaborn
    
    Parameters:
    - df: DataFrame with actual and predicted values
    - model_name: Name of the model
    
    Returns:
    - Matplotlib figure
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    sns.scatterplot(
        x=df.index, 
        y='Actual', 
        data=df, 
        color='blue', 
        label='Actual', 
        alpha=0.5
    )
    
    # Plot predicted values
    sns.scatterplot(
        x=df.index, 
        y='Predicted', 
        data=df, 
        color='red', 
        label='Predicted', 
        alpha=0.5
    )
    
    plt.xlabel('Index')
    plt.ylabel('CitedByCount')
    plt.title(f'Comparison of Actual vs Predicted CitedByCount - {model_name}')
    plt.legend()
    
    return plt

def main():
    # Set page configuration
    st.set_page_config(
        page_title='Model Predictions Visualization', 
        layout='wide'
    )
    
    # Title
    st.title('Model Predictions Comparison')
    
    # Results folder path (modify as needed)
    results_folder = 'result'
    
    # Load prediction files
    predictions = load_prediction_files(results_folder)
    
    # Sidebar for model selection
    selected_model = st.sidebar.selectbox(
        'Select Model', 
        list(predictions.keys())
    )
    
    # Main content area
    if selected_model:
        # Get selected model's predictions
        model_df = predictions[selected_model]
        
        # Create visualization
        fig = create_comparison_plot(model_df, selected_model)
        
        # Display plot
        st.pyplot(fig)
        
        # Display dataframe
        st.dataframe(model_df)

if __name__ == '__main__':
    main()