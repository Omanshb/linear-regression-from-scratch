import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from linear_from_scratch import (
    LinearRegressionFromScratch,
    GradientDescentLinearRegression,
    mse, mae, r2_score,
    standardize_features
)


def load_sample_datasets():
    """Load built-in regression datasets for demonstration."""
    datasets = {}
    
    try:
        from sklearn.datasets import fetch_california_housing
        california = fetch_california_housing()
        datasets['California Housing'] = {
            'data': pd.DataFrame(california.data, columns=california.feature_names),
            'target': california.target,
            'description': 'Median house values for California districts (1990 census)'
        }
    except ImportError:
        pass
    
    try:
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
        datasets['Diabetes'] = {
            'data': pd.DataFrame(diabetes.data, columns=diabetes.feature_names),
            'target': diabetes.target,
            'description': 'Diabetes disease progression dataset with 10 baseline variables'
        }
    except ImportError:
        pass
    
    try:
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)
        feature_names = [f'Feature_{i+1}' for i in range(5)]
        datasets['Synthetic Data'] = {
            'data': pd.DataFrame(X, columns=feature_names),
            'target': y,
            'description': 'Synthetic regression dataset with 5 features and controlled noise'
        }
    except ImportError:
        pass
    
    return datasets


def create_actual_vs_predicted_plot(y_true, y_pred, title):
    """Create scatter plot of actual vs predicted values."""
    df_plot = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    fig = px.scatter(
        df_plot, x='Actual', y='Predicted',
        title=title,
        labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'}
    )
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(width=700, height=500)
    return fig


def create_residual_plot(y_true, y_pred, title):
    """Create residual plot."""
    residuals = y_true - y_pred
    
    df_plot = pd.DataFrame({
        'Predicted': y_pred,
        'Residuals': residuals
    })
    
    fig = px.scatter(
        df_plot, x='Predicted', y='Residuals',
        title=title,
        labels={'Predicted': 'Predicted Values', 'Residuals': 'Residuals'}
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(width=700, height=500)
    return fig


def create_feature_importance_plot(coefficients, feature_names, title):
    """Create bar plot of feature coefficients."""
    df_plot = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    df_plot = df_plot.sort_values('Coefficient', key=abs, ascending=True)
    
    fig = px.bar(
        df_plot, x='Coefficient', y='Feature',
        orientation='h',
        title=title,
        labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature Name'}
    )
    
    fig.update_layout(width=700, height=max(400, len(feature_names) * 30))
    return fig


def create_cost_history_plot(cost_history, title):
    """Create line plot of cost history for gradient descent."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(cost_history))),
            y=cost_history,
            mode='lines',
            name='Cost',
            line=dict(color='blue')
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Iteration',
        yaxis_title='Mean Squared Error',
        width=700,
        height=500
    )
    
    return fig


def create_regression_line_plot(X, y, y_pred, feature_name, title):
    """Create scatter plot with regression line for single feature."""
    df_plot = pd.DataFrame({
        'Feature': X.flatten(),
        'Actual': y,
        'Predicted': y_pred
    })
    
    df_plot = df_plot.sort_values('Feature')
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df_plot['Feature'],
            y=df_plot['Actual'],
            mode='markers',
            name='Actual Data',
            marker=dict(color='blue', size=8)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_plot['Feature'],
            y=df_plot['Predicted'],
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=3)
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title=feature_name,
        yaxis_title='Target Value',
        width=700,
        height=500
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="Linear Regression from Scratch",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Linear Regression from Scratch")
    
    st.sidebar.header("Data Selection")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Built-in Datasets", "Upload CSV"]
    )
    
    df = None
    target = None
    dataset_name = ""
    
    if data_source == "Built-in Datasets":
        datasets = load_sample_datasets()
        
        if not datasets:
            st.error("No built-in datasets available. Please install scikit-learn.")
            return
        
        dataset_choice = st.sidebar.selectbox(
            "Select dataset:",
            list(datasets.keys())
        )
        
        if dataset_choice:
            dataset = datasets[dataset_choice]
            df = dataset['data']
            target = dataset['target']
            dataset_name = dataset_choice
            
            st.sidebar.info(f"**{dataset_choice}**\n\n{dataset['description']}")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name
            
            if len(df.columns) > 1:
                target_col = st.sidebar.selectbox(
                    "Select target column:",
                    list(df.columns)
                )
                
                if target_col:
                    target = df[target_col].values
                    df = df.drop(columns=[target_col])
    
    if df is not None and target is not None:
        st.header(f"Dataset: {dataset_name}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Target Range", f"{target.min():.2f} - {target.max():.2f}")
        
        with st.expander("Dataset Preview"):
            preview_df = df.copy()
            preview_df['Target'] = target
            st.dataframe(preview_df.head(10))
        
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type:",
                ["Ordinary Least Squares (OLS)", "Gradient Descent"]
            )
        
        with col2:
            feature_selection = st.multiselect(
                "Select features (leave empty for all):",
                list(df.columns),
                default=[]
            )
        
        if len(feature_selection) == 0:
            feature_selection = list(df.columns)
        
        X = df[feature_selection].values
        y = target
        
        from sklearn.model_selection import train_test_split
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        model_params = {}
        
        if model_type == "Gradient Descent":
            st.markdown("#### Gradient Descent Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=1.0,
                    value=0.01,
                    step=0.001,
                    format="%.4f"
                )
            
            with col2:
                max_iterations = st.number_input(
                    "Max Iterations",
                    min_value=10,
                    max_value=10000,
                    value=1000,
                    step=100
                )
            
            with col3:
                batch_type = st.selectbox(
                    "Gradient Descent Type:",
                    ["Batch", "Mini-Batch", "Stochastic"]
                )
            
            if batch_type == "Mini-Batch":
                batch_size = st.slider(
                    "Batch Size",
                    min_value=2,
                    max_value=min(len(X_train), 100),
                    value=32
                )
            elif batch_type == "Stochastic":
                batch_size = 1
            else:
                batch_size = None
            
            model_params = {
                'learning_rate': learning_rate,
                'max_iterations': max_iterations,
                'batch_size': batch_size
            }
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                if model_type == "Ordinary Least Squares (OLS)":
                    model = LinearRegressionFromScratch(fit_intercept=True)
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    X_train_use = X_train
                    X_test_use = X_test
                else:
                    X_train_scaled, mean, std = standardize_features(X_train)
                    X_test_scaled = (X_test - mean) / std
                    
                    model = GradientDescentLinearRegression(
                        learning_rate=model_params['learning_rate'],
                        max_iterations=model_params['max_iterations'],
                        batch_size=model_params['batch_size'],
                        fit_intercept=True
                    )
                    model.fit(X_train_scaled, y_train, verbose=False)
                    
                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                    X_train_use = X_train_scaled
                    X_test_use = X_test_scaled
                
                st.session_state['model'] = model
                st.session_state['X_train'] = X_train_use
                st.session_state['X_test'] = X_test_use
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['y_train_pred'] = y_train_pred
                st.session_state['y_test_pred'] = y_test_pred
                st.session_state['feature_names'] = feature_selection
                st.session_state['model_type'] = model_type
        
        if 'model' in st.session_state:
            model = st.session_state['model']
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            y_train_pred = st.session_state['y_train_pred']
            y_test_pred = st.session_state['y_test_pred']
            feature_names = st.session_state['feature_names']
            model_type = st.session_state['model_type']
            
            st.header("Model Results")
            
            train_mse = mse(y_train, y_train_pred)
            train_mae = mae(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            
            test_mse = mse(y_test, y_test_pred)
            test_mae = mae(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Metrics")
                met_col1, met_col2, met_col3 = st.columns(3)
                with met_col1:
                    st.metric("MSE", f"{train_mse:.4f}")
                with met_col2:
                    st.metric("MAE", f"{train_mae:.4f}")
                with met_col3:
                    st.metric("R²", f"{train_r2:.4f}")
            
            with col2:
                st.subheader("Test Metrics")
                met_col1, met_col2, met_col3 = st.columns(3)
                with met_col1:
                    st.metric("MSE", f"{test_mse:.4f}")
                with met_col2:
                    st.metric("MAE", f"{test_mae:.4f}")
                with met_col3:
                    st.metric("R²", f"{test_r2:.4f}")
            
            st.markdown("---")
            
            with st.expander("Model Coefficients"):
                st.write(f"**Intercept:** {model.intercept_:.4f}")
                st.write("**Coefficients:**")
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': model.coefficients_
                })
                st.dataframe(coef_df)
            
            st.subheader("Visualizations")
            
            tab_names = ["Actual vs Predicted", "Residual Plot", "Feature Importance"]
            
            if model_type == "Gradient Descent" and hasattr(model, 'cost_history_'):
                tab_names.append("Cost History")
            
            if len(feature_names) == 1:
                tab_names.append("Regression Line")
            
            viz_tabs = st.tabs(tab_names)
            
            with viz_tabs[0]:
                fig_train = create_actual_vs_predicted_plot(
                    y_train, y_train_pred,
                    "Training Set: Actual vs Predicted"
                )
                st.plotly_chart(fig_train, use_container_width=True)
                
                fig_test = create_actual_vs_predicted_plot(
                    y_test, y_test_pred,
                    "Test Set: Actual vs Predicted"
                )
                st.plotly_chart(fig_test, use_container_width=True)
                
                st.info("""
                **Actual vs Predicted Plot:**
                - Points close to the red dashed line indicate accurate predictions
                - Scatter above the line means over-prediction
                - Scatter below the line means under-prediction
                - Tighter clustering around the line indicates better model performance
                """)
            
            with viz_tabs[1]:
                fig_residual = create_residual_plot(
                    y_test, y_test_pred,
                    "Residual Plot (Test Set)"
                )
                st.plotly_chart(fig_residual, use_container_width=True)
                
                st.info("""
                **Residual Plot:**
                - Residuals should be randomly scattered around zero
                - Patterns in residuals suggest the model is missing something
                - Funnel shapes indicate heteroscedasticity (non-constant variance)
                - Random scatter indicates good model fit
                """)
            
            with viz_tabs[2]:
                fig_importance = create_feature_importance_plot(
                    model.coefficients_,
                    feature_names,
                    "Feature Coefficients"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
                
                st.info("""
                **Feature Coefficients:**
                - Shows the impact of each feature on the target variable
                - Positive coefficients increase the target value
                - Negative coefficients decrease the target value
                - Larger absolute values indicate stronger influence
                """)
            
            current_tab_idx = 3
            
            if model_type == "Gradient Descent" and hasattr(model, 'cost_history_'):
                with viz_tabs[current_tab_idx]:
                    fig_cost = create_cost_history_plot(
                        model.cost_history_,
                        f"Cost History - {model.get_batch_type()}"
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
                    
                    st.info(f"""
                    **Cost History ({model.get_batch_type()}):**
                    - Shows how the model's error decreased during training
                    - Steep drops early on indicate rapid learning
                    - Flattening curve suggests convergence
                    - Oscillations may indicate learning rate is too high
                    - Final cost: {model.cost_history_[-1]:.4f}
                    """)
                current_tab_idx += 1
            
            if len(feature_names) == 1:
                with viz_tabs[current_tab_idx]:
                    fig_line = create_regression_line_plot(
                        X_test, y_test, y_test_pred,
                        feature_names[0],
                        f"Regression Line - {feature_names[0]}"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                    
                    st.info("""
                    **Regression Line:**
                    - Blue points show actual data
                    - Red line shows the fitted regression line
                    - The line represents the model's learned relationship
                    - Distance from points to line shows prediction error
                    """)
    
    else:
        st.info("Please select a data source from the sidebar to get started!")


if __name__ == "__main__":
    main()

