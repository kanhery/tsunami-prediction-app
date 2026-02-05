# 🌊 Earthquake Tsunami Prediction System

A comprehensive machine learning application for predicting tsunami occurrences following earthquake events. This project demonstrates binary classification using 6 different ML algorithms deployed on Streamlit Cloud.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Results](#results)

## 🎯 Overview

This application is part of **ML Assignment 2** and showcases:
- Implementation of 6 different classification algorithms
- Comprehensive model performance comparison
- Interactive web interface for tsunami prediction
- Data exploration and visualization
- Cloud deployment on Streamlit

## 📊 Dataset

**Name:** Earthquake Tsunami Prediction Dataset  
**Source:** Kaggle/Public Dataset  
**Samples:** 782  
**Features:** 12  
**Target:** Binary (Tsunami: Yes/No)

### Features:
1. `latitude` - Geographic latitude of earthquake epicenter
2. `longitude` - Geographic longitude of earthquake epicenter
3. `focal_depth` - Depth of earthquake focus (km)
4. `magnitude` - Earthquake magnitude
5. `cdi` - Community Decimal Intensity
6. `mmi` - Modified Mercalli Intensity
7. `alert` - Alert level (green/yellow/orange/red)
8. `sig` - Significance measure
9. `nst` - Number of seismic stations
10. `dmin` - Minimum distance to stations
11. `gap` - Azimuthal gap
12. `magtype` - Type of magnitude measurement

## 🤖 Models

The project implements and compares 6 machine learning algorithms:

1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Non-parametric tree-based classifier
3. **K-Nearest Neighbors (kNN)** - Instance-based learning
4. **Naive Bayes (Gaussian)** - Probabilistic classifier
5. **Random Forest** - Ensemble of decision trees (bagging)
6. **XGBoost** - Gradient boosting ensemble

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
