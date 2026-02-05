import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Earthquake Tsunami Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌊 Earthquake Tsunami Prediction System</h1>', unsafe_allow_html=True)
st.markdown("### Machine Learning Assignment 2 - Binary Classification")
st.markdown("---")

# Load models and data
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_names = ['logistic_regression', 'decision_tree', 'knn', 'naive_bayes', 'random_forest', 'xgboost']
    
    for name in model_names:
        model_path = Path(f'models/{name}_model.pkl')
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
    
    return models

@st.cache_data
def load_data():
    """Load earthquake tsunami training dataset for exploration"""
    # Use training data for data exploration page
    data = pd.read_csv('earthquake_tsunami_train.csv')
    return data

@st.cache_data
def load_results():
    """Load model results"""
    results_path = Path('models/model_results.csv')
    if results_path.exists():
        return pd.read_csv(results_path)
    return None

@st.cache_resource
def load_scaler():
    """Load the trained scaler"""
    scaler_path = Path('models/scaler.pkl')
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None

# Sidebar
with st.sidebar:
    st.header("📊 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "📈 Model Performance", "� Test Data Upload", "�🔮 Make Predictions", "📊 Data Exploration", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.info("""
    **Dataset:** Earthquake Tsunami Prediction  
    **Source:** Kaggle/Public Dataset  
    **Samples:** 782  
    **Features:** 12  
    **Task:** Binary Classification
    """)

# Page: Home
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to the Earthquake Tsunami Prediction System")
        st.markdown("""
        This application uses machine learning to predict the likelihood of a tsunami
        following an earthquake event. We've implemented and compared **6 different ML models**:
        
        1. **Logistic Regression** - Linear baseline model
        2. **Decision Tree** - Non-parametric tree-based classifier
        3. **K-Nearest Neighbors (kNN)** - Instance-based learning
        4. **Naive Bayes** - Probabilistic classifier
        5. **Random Forest** - Ensemble of decision trees
        6. **XGBoost** - Gradient boosting ensemble
        
        ### 🎯 Key Features
        - Compare multiple ML algorithms
        - Interactive model performance visualization
        - Real-time tsunami prediction
        - Comprehensive data exploration
        """)
    
    with col2:
        st.image("https://img.icons8.com/color/256/tsunami.png", width=200)
        
    st.markdown("---")
    
    # Load and display quick stats
    try:
        results = load_results()
        if results is not None:
            st.subheader("📊 Quick Performance Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_acc_model = results.loc[results['Accuracy'].idxmax(), 'Model']
                best_acc = results['Accuracy'].max()
                st.metric("Best Accuracy", f"{best_acc:.4f}", f"({best_acc_model})")
            
            with col2:
                best_auc_model = results.loc[results['AUC Score'].idxmax(), 'Model']
                best_auc = results['AUC Score'].max()
                st.metric("Best AUC Score", f"{best_auc:.4f}", f"({best_auc_model})")
            
            with col3:
                best_f1_model = results.loc[results['F1 Score'].idxmax(), 'Model']
                best_f1 = results['F1 Score'].max()
                st.metric("Best F1 Score", f"{best_f1:.4f}", f"({best_f1_model})")
    except:
        st.warning("⚠️ Model results not found. Please train the models first by running `python train_models.py`")

# Page: Model Performance
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Comparison")
    
    results = load_results()
    
    if results is None:
        st.error("❌ Model results not found. Please train the models first by running `python train_models.py`")
    else:
        # Display results table
        st.subheader("📋 Performance Metrics Table")
        st.dataframe(results.style.highlight_max(axis=0, subset=['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']), use_container_width=True)
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📊 Performance Visualizations")
        
        # Metric comparison charts
        metrics = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
        
        tab1, tab2, tab3 = st.tabs(["Bar Charts", "Radar Chart", "Heatmap"])
        
        with tab1:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Model Performance Comparison Across All Metrics', fontsize=16, fontweight='bold')
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for idx, (ax, metric) in enumerate(zip(axes.flatten(), metrics)):
                bars = ax.bar(results['Model'], results[metric], color=colors[idx], alpha=0.7)
                ax.set_title(metric, fontsize=12, fontweight='bold')
                ax.set_ylabel('Score', fontsize=10)
                ax.set_ylim([0, 1])
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                ax.grid(axis='y', alpha=0.3)
                
                # Highlight best model
                best_idx = results[metric].idxmax()
                bars[best_idx].set_color(colors[idx])
                bars[best_idx].set_alpha(1.0)
                bars[best_idx].set_edgecolor('black')
                bars[best_idx].set_linewidth(2)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            # Radar chart
            from math import pi
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
            angles += angles[:1]
            
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_rlabel_position(0)
            
            plt.xticks(angles[:-1], metrics, size=10)
            ax.set_ylim(0, 1)
            
            for idx, model_name in enumerate(results['Model']):
                values = results.loc[idx, metrics].values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
                ax.fill(angles, values, alpha=0.1)
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.title('Model Performance Radar Chart', size=14, fontweight='bold', pad=20)
            st.pyplot(fig)
        
        with tab3:
            # Heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            
            heatmap_data = results.set_index('Model')[metrics].T
            sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', 
                       cbar_kws={'label': 'Score'}, ax=ax, vmin=0, vmax=1)
            
            plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Metric', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Best model by metric
        st.subheader("🏆 Best Model by Metric")
        
        cols = st.columns(3)
        for idx, metric in enumerate(metrics):
            with cols[idx % 3]:
                best_idx = results[metric].idxmax()
                best_model = results.loc[best_idx, 'Model']
                best_score = results.loc[best_idx, metric]
                st.metric(metric, f"{best_score:.4f}", f"{best_model}")

# Page: Test Data Upload
elif page == "📤 Test Data Upload":
    st.header("📤 Upload Test Data for Evaluation")
    st.markdown("""
    Upload a CSV file containing test earthquake data to evaluate the trained models.
    The CSV should have the same features as the training data (without the 'tsunami' column).
    
    **Required Features:** latitude, longitude, focal_depth, magnitude, cdi, mmi, alert, sig, nst, dmin, gap, magtype
    """)
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            test_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(test_data)} samples")
            
            # Display sample
            st.subheader("📋 Data Preview")
            st.dataframe(test_data.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Check if target column exists
            if 'tsunami' in test_data.columns:
                st.subheader("🤖 Model Evaluation on Uploaded Test Data")
                
                # Prepare data
                X_test = test_data.drop('tsunami', axis=1)
                y_test = test_data['tsunami'].values
                
                # Encode categorical features if needed
                if 'alert' in X_test.columns and X_test['alert'].dtype == 'object':
                    alert_map = {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}
                    X_test['alert'] = X_test['alert'].map(alert_map).fillna(0)
                
                if 'magtype' in X_test.columns and X_test['magtype'].dtype == 'object':
                    magtype_map = {'mb': 0, 'ml': 1, 'ms': 2, 'mw': 3, 'md': 4}
                    X_test['magtype'] = X_test['magtype'].map(magtype_map).fillna(0)
                
                # Load scaler and scale features
                scaler = load_scaler()
                if scaler is not None:
                    X_test_scaled = scaler.transform(X_test)
                else:
                    st.warning("⚠️ Scaler not found. Using unscaled data.")
                    X_test_scaled = X_test.values
                
                # Model selection
                models = load_models()
                model_choice = st.selectbox(
                    "Select Model for Evaluation",
                    options=list(models.keys()),
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                if st.button("📊 Evaluate Model", type="primary"):
                    model = models[model_choice]
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    auc = roc_auc_score(y_test, y_pred_proba)
                    
                    # Display metrics
                    st.subheader("📊 Evaluation Metrics")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("Precision", f"{precision:.4f}")
                    with col3:
                        st.metric("Recall", f"{recall:.4f}")
                    with col4:
                        st.metric("F1 Score", f"{f1:.4f}")
                    with col5:
                        st.metric("AUC", f"{auc:.4f}")
                    
                    st.markdown("---")
                    
                    # Confusion Matrix
                    st.subheader("🎯 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['No Tsunami', 'Tsunami'],
                               yticklabels=['No Tsunami', 'Tsunami'],
                               cbar_kws={'label': 'Count'})
                    plt.title(f'Confusion Matrix - {model_choice.replace("_", " ").title()}', 
                             fontsize=14, fontweight='bold')
                    plt.ylabel('Actual', fontsize=12)
                    plt.xlabel('Predicted', fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.markdown("---")
                    st.subheader("📋 Classification Report")
                    
                    report = classification_report(y_test, y_pred, 
                                                  target_names=['No Tsunami', 'Tsunami'],
                                                  output_dict=True)
                    
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=1), 
                               use_container_width=True)
                    
                    # Additional insights
                    st.markdown("---")
                    st.subheader("📈 Additional Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Prediction distribution
                        fig, ax = plt.subplots(figsize=(8, 5))
                        pred_dist = pd.Series(y_pred).value_counts().sort_index()
                        colors = ['green', 'red']
                        ax.bar(['No Tsunami', 'Tsunami'], pred_dist.values, color=colors, alpha=0.7)
                        ax.set_title('Prediction Distribution', fontweight='bold')
                        ax.set_ylabel('Count')
                        for i, v in enumerate(pred_dist.values):
                            ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        # Probability distribution
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.hist(y_pred_proba, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                        ax.set_title('Prediction Probability Distribution', fontweight='bold')
                        ax.set_xlabel('Tsunami Probability')
                        ax.set_ylabel('Frequency')
                        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                    
            else:
                st.warning("⚠️ The uploaded CSV does not contain a 'tsunami' column. Please upload test data with labels for evaluation.")
                st.info("💡 If you want to make predictions on unlabeled data, use the **Make Predictions** page instead.")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format and column names.")
    else:
        st.info("👆 Please upload a CSV file to begin evaluation")
        
        # Show example format
        with st.expander("📝 View Expected CSV Format"):
            st.markdown("""
            Your CSV should have these columns:
            
            **Features (X):**
            - latitude, longitude, focal_depth, magnitude
            - cdi, mmi, alert, sig
            - nst, dmin, gap, magtype
            
            **Target (y):**
            - tsunami (0 = No Tsunami, 1 = Tsunami)
            
            **Example:**
            ```
            latitude,longitude,focal_depth,magnitude,cdi,mmi,alert,sig,nst,dmin,gap,magtype,tsunami
            35.5,-120.3,10.2,5.4,3.2,4.5,green,450,25,0.5,85,mw,0
            40.2,-125.1,15.8,6.8,5.1,6.2,orange,820,45,0.3,45,mw,1
            ```
            """)

# Page: Make Predictions
elif page == "🔮 Make Predictions":
    st.header("🔮 Predict Tsunami Likelihood")
    
    models = load_models()
    
    if not models:
        st.error("❌ Models not found. Please train the models first by running `python train_models.py`")
    else:
        st.markdown("Enter earthquake parameters to predict tsunami likelihood:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latitude = st.number_input("Latitude", value=0.0, format="%.4f")
            longitude = st.number_input("Longitude", value=0.0, format="%.4f")
            focal_depth = st.number_input("Focal Depth (km)", value=10.0, min_value=0.0)
            magnitude = st.number_input("Magnitude", value=5.0, min_value=0.0, max_value=10.0)
        
        with col2:
            cdi = st.number_input("CDI (Community Decimal Intensity)", value=0.0, min_value=0.0)
            mmi = st.number_input("MMI (Modified Mercalli Intensity)", value=0.0, min_value=0.0)
            alert = st.selectbox("Alert Level", options=['green', 'yellow', 'orange', 'red'])
            sig = st.number_input("Significance", value=0.0, min_value=0.0)
        
        with col3:
            nst = st.number_input("Number of Stations", value=0, min_value=0)
            dmin = st.number_input("Minimum Distance (degrees)", value=0.0, min_value=0.0)
            gap = st.number_input("Azimuthal Gap (degrees)", value=0.0, min_value=0.0)
            magtype = st.selectbox("Magnitude Type", options=['mb', 'ml', 'ms', 'mw', 'md'])
        
        st.markdown("---")
        
        # Select model
        model_choice = st.selectbox(
            "Select Model for Prediction",
            options=list(models.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if st.button("🔮 Predict", type="primary"):
            # Encode categorical features (simplified for demo)
            alert_map = {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}
            magtype_map = {'mb': 0, 'ml': 1, 'ms': 2, 'mw': 3, 'md': 4}
            
            # Create feature array
            features = np.array([[
                latitude, longitude, focal_depth, magnitude,
                cdi, mmi, alert_map.get(alert, 0), sig,
                nst, dmin, gap, magtype_map.get(magtype, 0)
            ]])
            
            # Make prediction
            model = models[model_choice]
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **TSUNAMI WARNING**")
                    st.markdown("### High Risk of Tsunami")
                else:
                    st.success("✅ **NO TSUNAMI EXPECTED**")
                    st.markdown("### Low Risk of Tsunami")
            
            with col2:
                st.metric("Tsunami Probability", f"{probability[1]:.2%}")
                st.metric("No Tsunami Probability", f"{probability[0]:.2%}")
            
            # Probability gauge
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(['No Tsunami', 'Tsunami'], probability, color=['green', 'red'])
            ax.set_xlabel('Probability')
            ax.set_xlim([0, 1])
            ax.set_title(f'Prediction Confidence - {model_choice.replace("_", " ").title()}')
            for i, v in enumerate(probability):
                ax.text(v + 0.02, i, f'{v:.2%}', va='center')
            plt.tight_layout()
            st.pyplot(fig)

# Page: Data Exploration
elif page == "📊 Data Exploration":
    st.header("📊 Dataset Exploration")
    
    data = load_data()
    
    st.subheader("🔍 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        st.metric("Features", len(data.columns) - 1)
    with col3:
        st.metric("Tsunami Cases", data['tsunami'].sum())
    with col4:
        st.metric("Non-Tsunami Cases", len(data) - data['tsunami'].sum())
    
    st.markdown("---")
    
    # Display data
    st.subheader("📋 Data Sample")
    st.dataframe(data.head(20), use_container_width=True)
    
    st.markdown("---")
    
    # Data statistics
    st.subheader("📈 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📊 Data Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Target Distribution", "Feature Distributions", "Correlations"])
    
    with tab1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Count plot
        tsunami_counts = data['tsunami'].value_counts()
        ax1.bar(['No Tsunami', 'Tsunami'], tsunami_counts.values, color=['green', 'red'], alpha=0.7)
        ax1.set_ylabel('Count')
        ax1.set_title('Target Variable Distribution')
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart
        ax2.pie(tsunami_counts.values, labels=['No Tsunami', 'Tsunami'], 
                autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
        ax2.set_title('Target Variable Proportion')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('tsunami')
        
        selected_feature = st.selectbox("Select Feature to Visualize", numeric_cols)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        ax1.hist(data[selected_feature], bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel(selected_feature)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{selected_feature} Distribution')
        ax1.grid(axis='y', alpha=0.3)
        
        # Box plot by target
        data.boxplot(column=selected_feature, by='tsunami', ax=ax2)
        ax2.set_xlabel('Tsunami')
        ax2.set_ylabel(selected_feature)
        ax2.set_title(f'{selected_feature} by Tsunami Status')
        plt.suptitle('')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        # Correlation heatmap
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)

# Page: About
elif page == "ℹ️ About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🌊 Earthquake Tsunami Prediction System
    
    ### Project Overview
    This application demonstrates a comprehensive machine learning approach to predicting
    tsunami occurrences following earthquake events. It is part of **ML Assignment 2** focusing
    on binary classification using multiple algorithms.
    
    ### 📚 Dataset Information
    - **Name:** Earthquake Tsunami Prediction Dataset
    - **Source:** Kaggle/Public Dataset
    - **Total Samples:** 782
    - **Number of Features:** 12
    - **Target Variable:** Binary (Tsunami: Yes/No)
    - **Problem Type:** Binary Classification
    
    ### 🔧 Features Used
    1. **latitude** - Geographic latitude of earthquake epicenter
    2. **longitude** - Geographic longitude of earthquake epicenter
    3. **focal_depth** - Depth of earthquake focus (km)
    4. **magnitude** - Earthquake magnitude
    5. **cdi** - Community Decimal Intensity
    6. **mmi** - Modified Mercalli Intensity
    7. **alert** - Alert level (green/yellow/orange/red)
    8. **sig** - Significance measure
    9. **nst** - Number of seismic stations
    10. **dmin** - Minimum distance to stations
    11. **gap** - Azimuthal gap
    12. **magtype** - Type of magnitude measurement
    
    ### 🤖 Machine Learning Models
    
    1. **Logistic Regression**
       - Linear baseline model
       - Fast training and prediction
       - Interpretable coefficients
    
    2. **Decision Tree**
       - Non-parametric approach
       - Captures non-linear patterns
       - Provides interpretable rules
    
    3. **K-Nearest Neighbors (kNN)**
       - Instance-based learning
       - No explicit training phase
       - Sensitive to feature scaling
    
    4. **Naive Bayes (Gaussian)**
       - Probabilistic classifier
       - Assumes feature independence
       - Fast and efficient
    
    5. **Random Forest**
       - Ensemble of decision trees
       - Reduces overfitting
       - Robust performance
    
    6. **XGBoost**
       - Gradient boosting ensemble
       - State-of-the-art performance
       - Includes regularization
    
    ### 📊 Evaluation Metrics
    - **Accuracy** - Overall correctness
    - **AUC Score** - Area under ROC curve
    - **Precision** - Positive prediction accuracy
    - **Recall** - True positive rate
    - **F1 Score** - Harmonic mean of precision and recall
    - **MCC Score** - Matthews Correlation Coefficient
    
    ### 🚀 Deployment
    This application is deployed on **Streamlit Cloud** for easy access and demonstration.
    
    ### 👨‍💻 Assignment Details
    - **Course:** Machine Learning
    - **Assignment:** ML Assignment 2
    - **Focus:** Binary Classification with Multiple Algorithms
    - **Platform:** Streamlit Cloud
    
    ### 📧 Contact
    For questions or feedback about this project, please refer to the course materials.
    
    ---
    
    **Note:** This is an educational project demonstrating ML classification techniques.
    Actual tsunami prediction systems require more sophisticated models and real-time data integration.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>🌊 Earthquake Tsunami Prediction System | ML Assignment 2 | Streamlit Cloud Deployment</p>
        <p>Built with ❤️ using Streamlit and Scikit-learn</p>
    </div>
    """,
    unsafe_allow_html=True
)
