# 🌊 Earthquake Tsunami Prediction System

**ML Assignment 2 - Binary Classification**  
**Streamlit Cloud Deployment**

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Models Used](#models-used)
- [Model Performance Comparison](#model-performance-comparison)
- [Observations on Model Performance](#observations-on-model-performance)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)

---

## 🎯 Problem Statement

Tsunamis are devastating natural disasters that can cause massive destruction to coastal areas, resulting in loss of life and property. Early prediction and warning systems are crucial for disaster preparedness and mitigation. 

**Objective:** Develop a machine learning-based binary classification system to predict the likelihood of a tsunami occurrence following an earthquake event. 

**Challenge:** Given various earthquake parameters (magnitude, depth, location, intensity measures, etc.), accurately classify whether a tsunami will occur (1) or not occur (0).

**Approach:** 
- Implement and compare 6 different machine learning algorithms
- Evaluate models using multiple performance metrics
- Deploy the best-performing models in an interactive web application
- Enable real-time predictions and comprehensive model evaluation

**Impact:** This system can assist:
- Early warning systems for coastal communities
- Emergency response planning and resource allocation
- Risk assessment and disaster preparedness strategies
- Scientific research on earthquake-tsunami relationships

---

## 📊 Dataset Description

### Overview
- **Dataset Name:** Earthquake Tsunami Prediction Dataset
- **Source:** Kaggle/Public Dataset
- **Total Samples:** 782 earthquake events
- **Training Set:** 625 samples (79.9%)
- **Test Set:** 157 samples (20.1%)
- **Number of Features:** 12
- **Target Variable:** `tsunami` (Binary: 0 = No Tsunami, 1 = Tsunami)
- **Problem Type:** Supervised Binary Classification
- **Class Distribution:**
  - No Tsunami (Class 0): 478 samples (61.1%)
  - Tsunami (Class 1): 304 samples (38.9%)

### Features Description

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| **latitude** | Numeric (Float) | Geographic latitude of earthquake epicenter | -90° to +90° |
| **longitude** | Numeric (Float) | Geographic longitude of earthquake epicenter | -180° to +180° |
| **focal_depth** | Numeric (Float) | Depth of earthquake focus below Earth's surface (km) | 0 to 700+ km |
| **magnitude** | Numeric (Float) | Earthquake magnitude (Richter scale or moment magnitude) | 0 to 10 |
| **cdi** | Numeric (Float) | Community Decimal Intensity - perceived shaking intensity | 0 to 10 |
| **mmi** | Numeric (Float) | Modified Mercalli Intensity - measure of earthquake effects | 0 to 12 |
| **alert** | Categorical | USGS alert level indicating severity | green, yellow, orange, red |
| **sig** | Numeric (Integer) | Significance value - impact measure | 0 to 2000+ |
| **nst** | Numeric (Integer) | Number of seismic stations reporting | 0 to 500+ |
| **dmin** | Numeric (Float) | Minimum distance to nearest seismic station (degrees) | 0 to 180° |
| **gap** | Numeric (Float) | Azimuthal gap - largest angle between stations (degrees) | 0 to 360° |
| **magtype** | Categorical | Type of magnitude measurement | mb, ml, ms, mw, md |

### Data Characteristics
- **No missing values:** Complete dataset with all features populated
- **Balanced dataset:** Reasonable distribution between classes
- **Mixed features:** Combination of numerical and categorical features
- **Geographical coverage:** Global earthquake events
- **Temporal relevance:** Historical earthquake-tsunami pairs

### Data Preprocessing
1. **Train-Test Split:** 80-20 stratified split maintaining class distribution
2. **Feature Scaling:** StandardScaler normalization for numerical features
3. **Categorical Encoding:** Label encoding for `alert` and `magtype`
4. **No feature selection:** All 12 features retained for modeling

---

## 🤖 Models Used

The project implements and evaluates **6 different machine learning algorithms** for binary classification:

1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Non-parametric tree-based classifier
3. **K-Nearest Neighbors (kNN)** - Instance-based learning
4. **Naive Bayes (Gaussian)** - Probabilistic classifier
5. **Random Forest** - Ensemble of decision trees (bagging)
6. **XGBoost** - Gradient boosting ensemble

### Model Details

#### 1. Logistic Regression
- **Type:** Linear Model
- **Parameters:** max_iter=1000, random_state=42
- **Strengths:** Fast, interpretable, probabilistic outputs
- **Use Case:** Baseline model

#### 2. Decision Tree
- **Type:** Tree-based Model  
- **Parameters:** random_state=42
- **Strengths:** Interpretable rules, handles mixed features
- **Use Case:** Feature importance analysis

#### 3. K-Nearest Neighbors (kNN)
- **Type:** Instance-based Learning
- **Parameters:** n_neighbors=5
- **Strengths:** No training phase, captures local patterns
- **Use Case:** Complex non-linear patterns

#### 4. Naive Bayes
- **Type:** Probabilistic Classifier
- **Parameters:** Gaussian distribution
- **Strengths:** Fast, efficient, small datasets
- **Use Case:** Quick baseline with probabilistic interpretation

#### 5. Random Forest (Ensemble)
- **Type:** Bagging Ensemble
- **Parameters:** n_estimators=100, random_state=42
- **Strengths:** Reduces overfitting, robust, feature interactions
- **Use Case:** High-accuracy predictions

#### 6. XGBoost (Ensemble)
- **Type:** Gradient Boosting
- **Parameters:** random_state=42, eval_metric='logloss'
- **Strengths:** State-of-the-art, regularization, feature importance
- **Use Case:** Maximum predictive accuracy

---

## 📊 Model Performance Comparison

All models were trained on the **training set (625 samples)** and evaluated on the **test set (157 samples)** using stratified 80-20 split.

### Performance Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| **Logistic Regression** | 0.8599 | 0.9319 | 0.7746 | 0.9016 | 0.8333 | 0.7198 |
| **Decision Tree** | 0.8854 | 0.8734 | 0.8772 | 0.8197 | 0.8475 | 0.7569 |
| **kNN** | 0.8854 | 0.9258 | 0.8209 | 0.9016 | 0.8594 | 0.7654 |
| **Naive Bayes** | 0.8280 | 0.8613 | 0.7237 | 0.9016 | 0.8029 | 0.6660 |
| **Random Forest (Ensemble)** | **0.9299** | 0.9641 | 0.8676 | **0.9672** | **0.9147** | **0.8592** |
| **XGBoost (Ensemble)** | 0.9236 | **0.9679** | **0.8889** | 0.9180 | 0.9032 | 0.8404 |

**Metric Definitions:**
- **Accuracy:** Proportion of correct predictions (TP+TN)/(TP+TN+FP+FN)
- **AUC:** Area Under ROC Curve - model's ability to distinguish classes
- **Precision:** Proportion of true positives among positive predictions TP/(TP+FP)
- **Recall:** Proportion of actual positives correctly identified TP/(TP+FN)
- **F1 Score:** Harmonic mean of precision and recall
- **MCC:** Matthews Correlation Coefficient - balanced measure

### Best Model by Metric

| Metric | Best Model | Score |
|--------|-----------|-------|
| **Accuracy** | Random Forest | 0.9299 (92.99%) |
| **AUC Score** | XGBoost | 0.9679 (96.79%) |
| **Precision** | XGBoost | 0.8889 (88.89%) |
| **Recall** | Random Forest | 0.9672 (96.72%) |
| **F1 Score** | Random Forest | 0.9147 (91.47%) |
| **MCC Score** | Random Forest | 0.8592 |

---

## 📝 Observations on Model Performance

### Detailed Analysis of Each Model

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Achieved solid baseline performance with 85.99% accuracy and impressive AUC of 0.9319. The model shows good recall (90.16%) meaning it captures most tsunami events, but relatively lower precision (77.46%) indicates some false alarms. As a linear model, it's fast and interpretable but may miss complex non-linear patterns in earthquake-tsunami relationships. Best suited as a quick baseline or when model interpretability is critical. |
| **Decision Tree** | Demonstrated good performance with 88.54% accuracy and balanced precision-recall trade-off. The model achieved high precision (87.72%), making fewer false tsunami predictions, but moderate recall (81.97%). Its tree structure provides excellent interpretability, showing clear decision rules based on earthquake parameters. However, the relatively lower AUC (0.8734) suggests potential overfitting to training data. Useful when understanding feature importance and decision paths is crucial. |
| **kNN** | Achieved 88.54% accuracy with strong AUC of 0.9258, showing good discriminative ability between classes. The model benefits from capturing local patterns in the feature space with k=5 neighbors. Good recall (90.16%) ensures most tsunamis are detected, while precision (82.09%) is acceptable. Being instance-based, it requires storing all training data and can be slow for predictions. Performance depends heavily on feature scaling, which was properly applied. |
| **Naive Bayes** | Showed the lowest overall performance with 82.80% accuracy, reflecting the limitations of its feature independence assumption. Despite high recall (90.16%) - detecting most tsunamis - it suffers from low precision (72.37%), producing many false alarms. The lowest AUC (0.8613) and MCC (0.6660) indicate weaker overall predictive power. However, it's extremely fast and memory-efficient, making it suitable for real-time applications where speed trumps accuracy. The poor performance suggests strong feature dependencies in earthquake-tsunami data. |
| **Random Forest (Ensemble)** | **Outstanding performance as the top model** with 92.99% accuracy and highest F1 score (0.9147). Exceptional recall of 96.72% means it catches nearly all tsunami events with minimal misses - critical for disaster warning systems. The ensemble of 100 trees effectively captures complex non-linear patterns and feature interactions. High MCC (0.8592) indicates excellent performance across all confusion matrix quadrants. The model balances precision and recall optimally, making it the recommended choice for deployment. Robust to overfitting and provides reliable feature importance rankings. |
| **XGBoost (Ensemble)** | Achieved near-top performance with 92.36% accuracy and the **highest AUC (0.9679)** and **precision (88.89%)**. This means when it predicts a tsunami, it's highly reliable with fewer false alarms compared to other models. The gradient boosting approach with regularization prevents overfitting while capturing complex patterns. Excellent MCC (0.8404) demonstrates balanced performance. Slightly lower recall (91.80%) compared to Random Forest means it might miss a few more tsunami events, but its superior precision makes it valuable when minimizing false alarms is important. Second-best overall model. |

### Key Insights

1. **Ensemble Models Dominate:** Random Forest and XGBoost significantly outperform traditional algorithms, demonstrating the power of ensemble methods for complex pattern recognition.

2. **Recall is Critical:** For tsunami prediction, high recall (minimizing false negatives) is crucial since missing an actual tsunami has catastrophic consequences. Random Forest's 96.72% recall makes it ideal.

3. **Precision-Recall Trade-off:** XGBoost offers better precision (fewer false alarms) while Random Forest maximizes recall (fewer missed tsunamis). Choice depends on operational priorities.

4. **Feature Complexity:** The performance gap between linear (Logistic Regression) and non-linear models suggests complex, non-linear relationships between earthquake parameters and tsunami occurrence.

5. **Model Reliability:** All models achieve AUC > 0.86, indicating good discriminative ability. The high AUC scores (especially 0.96+) demonstrate reliable probability estimates for decision-making.

6. **Deployment Recommendation:** **Random Forest** for production deployment due to best overall balance of metrics, especially critical high recall for disaster prevention.

---

## ✨ Features

- **Model Performance Comparison**: Visualize and compare all 6 models across multiple metrics
- **Interactive Predictions**: Make real-time tsunami predictions with custom inputs
- **Data Exploration**: Comprehensive dataset analysis with interactive visualizations
- **Multiple Visualizations**: Bar charts, radar charts, heatmaps, and distribution plots
- **Responsive Design**: Clean, modern UI that works on all devices

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

1. **Clone or download this repository**

2. **Navigate to the project directory**
```bash
cd ML
```

3. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Ensure dataset is in the correct location**
```bash
# The earthquake_data_tsunami.csv file should be in the root directory
ls earthquake_data_tsunami.csv
```

## 📖 Usage

### Training Models

Before running the Streamlit app, train all models:

```bash
python train_models.py
```

This script will:
- Load and preprocess the dataset
- Train all 6 models
- Save trained models to the `models/` directory
- Generate performance metrics and save to `models/model_results.csv`
- Display performance comparison

### Running the Streamlit App

Launch the web application locally:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Home Page**: Overview and quick performance metrics
2. **Model Performance**: Detailed comparison with visualizations
3. **Make Predictions**: Input earthquake parameters for tsunami prediction
4. **Data Exploration**: Explore dataset statistics and visualizations
5. **About**: Detailed project information

## ☁️ Deployment on Streamlit Cloud

### Option 1: Deploy from GitHub

1. **Push your code to GitHub**
```bash
git init
git add .
git commit -m "Initial commit - ML Assignment 2"
git remote add origin <your-repo-url>
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`
   - Click "Deploy"

### Option 2: Deploy directly

1. **Ensure all files are ready**
   - `app.py` - Main Streamlit application
   - `train_models.py` - Model training script
   - `requirements.txt` - Dependencies
   - `earthquake_data_tsunami.csv` - Dataset
   - `models/` directory (create it if not exists)

2. **Run training locally first**
```bash
python train_models.py
```

3. **Commit models directory**
```bash
git add models/
git commit -m "Add trained models"
git push
```

4. **Streamlit Cloud will automatically rebuild and deploy**

### Important Notes for Deployment:

- **Dataset**: Ensure `earthquake_data_tsunami.csv` is in the root directory
- **Models**: The `models/` directory with trained models must be included
- **Python Version**: Streamlit Cloud uses Python 3.9 by default
- **Dependencies**: All packages in `requirements.txt` will be installed automatically

## 📁 Project Structure

```
ML/
├── app.py                          # Main Streamlit application
├── train_models.py                 # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── earthquake_data_tsunami.csv     # Dataset
├── models/                         # Trained models directory
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl                  # StandardScaler for preprocessing
│   └── model_results.csv           # Performance metrics
├── data/                           # Data directory (optional)
└── notebooks/                      # Jupyter notebooks (optional)
    └── MLAssngment2.ipynb          # Original development notebook
```

## 📊 Results

### Evaluation Metrics

All models are evaluated using 6 comprehensive metrics:

- **Accuracy**: Overall prediction correctness
- **AUC Score**: Area under the ROC curve
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate (sensitivity)
- **F1 Score**: Harmonic mean of precision and recall
- **MCC Score**: Matthews Correlation Coefficient

### Model Performance

Results are automatically generated after training and can be viewed in:
- Terminal output from `train_models.py`
- `models/model_results.csv` file
- Streamlit app "Model Performance" page

## 🔧 Troubleshooting

### Common Issues

**1. Module not found error**
```bash
pip install -r requirements.txt --upgrade
```

**2. Dataset not found**
Ensure `earthquake_data_tsunami.csv` is in the project root directory

**3. Models not found in Streamlit app**
Run `python train_models.py` before launching the app

**4. Port already in use**
```bash
streamlit run app.py --server.port 8502
```

**5. Memory issues on Streamlit Cloud**
Ensure models are optimized and cached using `@st.cache_resource`

## 📝 Assignment Requirements Completed

- ✅ Multiple ML algorithms implemented (6 models)
- ✅ Comprehensive evaluation metrics
- ✅ Model performance comparison
- ✅ Interactive web application
- ✅ Data visualization
- ✅ Cloud deployment ready
- ✅ Complete documentation
- ✅ Requirements.txt for dependency management
- ✅ Clean project structure

## 🎓 Educational Purpose

This project is developed for educational purposes as part of ML Assignment 2. It demonstrates:
- Binary classification techniques
- Model comparison methodology
- Ensemble learning
- Web application development
- Cloud deployment practices

## 📧 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review Streamlit documentation: https://docs.streamlit.io
3. Check scikit-learn documentation: https://scikit-learn.org

## 📄 License

This project is created for educational purposes as part of ML Assignment 2.

## 🙏 Acknowledgments

- Dataset: Kaggle/Public Dataset
- Framework: Streamlit
- ML Libraries: scikit-learn, XGBoost
- Platform: Streamlit Cloud

---

**Note**: This is an educational project demonstrating ML classification techniques. Actual tsunami prediction systems require more sophisticated models, real-time data integration, and domain expertise.

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train_models.py

# Run locally
streamlit run app.py

# Deploy to Streamlit Cloud
# Push to GitHub and connect via share.streamlit.io
```

---

**ML Assignment 2** | Binary Classification | Streamlit Cloud Deployment
