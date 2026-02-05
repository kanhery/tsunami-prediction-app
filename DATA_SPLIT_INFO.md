# 📊 Dataset Split Information

## Overview

The earthquake tsunami dataset has been split into separate **train** and **test** CSV files as required by the assignment PDF:

> *"As streamlit free tier has limited capacity, upload only test data"*

## Files Created

### 1. **earthquake_tsunami_train.csv**
- **Samples:** 625 (79.9% of total data)
- **Size:** 34 KB
- **Purpose:** Training the 6 ML models
- **Distribution:**
  - No Tsunami (0): 382 samples (61.1%)
  - Tsunami (1): 243 samples (38.9%)
- **Used by:** `train_models.py`

### 2. **earthquake_tsunami_test.csv**
- **Samples:** 157 (20.1% of total data)
- **Size:** 8.5 KB
- **Purpose:** Testing and evaluation via Streamlit app upload
- **Distribution:**
  - No Tsunami (0): 96 samples (61.1%)
  - Tsunami (1): 61 samples (38.9%)
- **Used by:** Upload in app's "📤 Test Data Upload" page

## Split Strategy

- **Method:** Stratified split using `train_test_split()`
- **Test Size:** 20% (0.2)
- **Random State:** 42 (for reproducibility)
- **Stratification:** Based on `tsunami` column to maintain class balance

```python
train_data, test_data = train_test_split(
    data, 
    test_size=0.2, 
    random_state=42, 
    stratify=data['tsunami']
)
```

## Why This Approach?

### ✅ Advantages:

1. **Deployment Friendly**
   - Test file is only 8.5 KB (suitable for Streamlit free tier)
   - Can upload test data without exceeding memory limits

2. **Realistic Evaluation**
   - Models never see test data during training
   - True performance measurement on unseen data

3. **Reproducible**
   - Same split used in `train_models.py` and for app testing
   - Random state ensures consistent splits

4. **Assignment Compliant**
   - Follows PDF requirement: "upload only test data"
   - Separate files for train/test as specified

## How to Use

### For Training:
```bash
# Train models on training data
python train_models.py

# This reads: earthquake_tsunami_train.csv
# Outputs: models/*.pkl files with performance on test set
```

### For Testing in Streamlit:
```bash
# Run the app
streamlit run app.py

# Navigate to: 📤 Test Data Upload page
# Upload: earthquake_tsunami_test.csv
# Select model and evaluate
```

## Data Features (12 columns)

| Feature | Type | Description |
|---------|------|-------------|
| latitude | float | Latitude of earthquake epicenter |
| longitude | float | Longitude of earthquake epicenter |
| focal_depth | float | Depth of earthquake focus (km) |
| magnitude | float | Earthquake magnitude |
| cdi | float | Community Decimal Intensity |
| mmi | float | Modified Mercalli Intensity |
| alert | string | Alert level (green/yellow/orange/red) |
| sig | float | Significance value |
| nst | int | Number of seismic stations |
| dmin | float | Minimum distance to stations |
| gap | float | Azimuthal gap |
| magtype | string | Magnitude type (mb/ml/ms/mw/md) |

**Target:** `tsunami` (0 or 1)

## Verification

To verify the split:

```bash
cd /Users/kanhery.dube/Documents/mywork/ML

# Check file sizes
ls -lh earthquake_tsunami*.csv

# Count samples
wc -l earthquake_tsunami_train.csv  # 626 lines (625 + header)
wc -l earthquake_tsunami_test.csv   # 158 lines (157 + header)
```

## Model Performance on This Split

Using the train/test split, the models achieved:

| Model | Accuracy | AUC | F1 Score |
|-------|----------|-----|----------|
| Random Forest | 92.99% | 96.41% | 91.47% |
| XGBoost | 92.36% | 96.79% | 90.32% |
| KNN | 88.54% | 92.58% | 85.94% |
| Decision Tree | 88.54% | 87.34% | 84.75% |
| Logistic Regression | 85.99% | 93.19% | 83.33% |
| Naive Bayes | 82.80% | 86.13% | 80.29% |

These results are saved in `models/model_results.csv`.

## Deployment Notes

### For GitHub:
```bash
git add earthquake_tsunami_train.csv
git add earthquake_tsunami_test.csv
git commit -m "Add train/test split datasets"
```

### For Streamlit Cloud:
- Both files will be deployed with the app
- Training data used by app for "Data Exploration" page
- Test data uploaded by user for model evaluation
- Total deployment size: ~42.5 KB (both files)

## Testing Workflow

1. **Local Development:**
   ```bash
   python train_models.py  # Train on train.csv
   streamlit run app.py     # Test with upload
   ```

2. **Upload Test File:**
   - Go to "📤 Test Data Upload" page
   - Click "Browse files"
   - Select `earthquake_tsunami_test.csv`
   - Choose model from dropdown
   - Click "📊 Evaluate Model"

3. **View Results:**
   - Evaluation metrics (Accuracy, Precision, Recall, F1, AUC)
   - Confusion matrix heatmap
   - Classification report
   - Prediction distributions
   - Probability histogram

## Summary

✅ **Training:** 625 samples in `earthquake_tsunami_train.csv`  
✅ **Testing:** 157 samples in `earthquake_tsunami_test.csv`  
✅ **Same Split:** Used in both training script and app evaluation  
✅ **Assignment Compliant:** Separate files for train/test as required  
✅ **Lightweight:** Test file only 8.5 KB for cloud deployment  

---

**Created:** February 5, 2026  
**Split Method:** Stratified 80/20 split (random_state=42)  
**Status:** ✅ Ready for deployment
