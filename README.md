# 🏥 Health Status Classification Model 📊

## 🚀 1. Introduction

### 🎯 1.1 Objective of the Study
- **Goal**: Develop an optimal machine learning model to classify patients' health status (`health_status`) using comprehensive healthcare data.
- **Target Variable**:
  - `0` (**Normal**): Healthy condition
  - `1` (**Abnormal**): Unhealthy condition

---

## 🛠️ 2. Data Preprocessing and Feature Selection

### 📚 2.1 Dataset Overview and Missing Value Check
- **Dataset Overview**:
  - **Total Samples**: 140,000
  - **Variables**: 26 (combination of continuous and categorical)

| **Variable Name**    | **Description**                      |
|----------------------|--------------------------------------|
| `index`              | Unique index                         |
| `name`               | Patient name                         |
| `birth`              | Year of birth                        |
| `bp_high`, `bp_low`  | Blood pressure values                |
| `chol_good`, `chol_bad` | Good/Bad cholesterol levels      |
| `iron_lvl`, `creatinine` | Iron/Creatinine levels           |
| `urine_marker`      | Urine marker values                  |
| `sugar_lvl`, `lipid_lvl`, `fat_content` | Sugar, lipid, and fat levels |
| `oral_issues`       | Oral health issues                   |
| `stature_inch`, `mass_kg`, `midsection_inch` | Height, weight, and waist size |
| `immune_index`      | Immunity index                       |
| `enzyme_1`, `enzyme_2`, `enzyme_3` | Enzyme levels           |
| `vision_l`, `vision_r` | Left/Right vision                |
| `audio_l`, `audio_r` | Left/Right hearing test results      |
| `health_status`     | Health status (target variable)      |

- **Missing Value Check**:
  - **Result**: No missing values detected (`train_df.isna().sum()`).

---

### 🔍 2.2 Feature Importance Evaluation Using Mutual Information
- **Purpose**: Identify and eliminate less relevant features to enhance model performance.
- **Evaluation Criteria**: F1 Score and AUC Score using the LightGBM (LGBM) model.
- **Removal Condition**: Remove features if both F1 and AUC scores improve upon removal.
- **Retention Condition**: Keep features if scores remain constant or degrade.

---

### ❌ 2.3 Removed Features
- **Continuous Variables**:
  - `immune_index`: 
    - **Improvement**: F1 Score → 0.72401, AUC Score → 0.844841
  - `enzyme_1`: 
    - **Improvement**: F1 Score → 0.723322, AUC Score → 0.844418
- **Categorical Variables**:
  - `urine_marker`, `audio_l`, `audio_r`, `name`, `index`

---

### ➕ 2.4 Generated Features and Validation
- **Created Features**:
  - **`age`**: Calculated from year of birth.  
    `age = 2024 – birth`
  - **`BMI`**: Calculated using height and weight.  
    `BMI = mass_kg / (stature_inch × 0.0254)^2`
- **Validation Results**:
  - **Adding `age`**: F1 Score → 0.72686, AUC Score → 0.847056
  - **Adding `BMI`**: F1 Score → 0.730212, AUC Score → 0.847312

---

## 🤖 3. Model Building and Optimization

### ⚡ 3.1 Overview of LightGBM Model
**LightGBM** is a high-performance gradient boosting framework. For this project, the **DART (Dropout Additive Regression Trees)** boosting method was employed.

- **DART**:
  - **Technique**: Applies dropout (from neural networks) to tree-based models to mitigate overfitting.
  - **Mechanism**: Omits entire trees during training rather than individual features.
  - **Advantages**:
    - Reduces overfitting.
    - Enhances generalization performance.

---

### 🔀 3.2 Data Splitting (Using Stratified Sampling)
- **Split Ratio**: 90% Training | 10% Validation
- **Method**: **Stratified Sampling** to maintain target variable (`health_status`) distribution.

#### 📌 Why Stratified Sampling?
- **Ensures**: Preservation of class distribution ratios.
- **Benefits**:
  - Addresses class imbalance.
  - Provides stable and reliable evaluation metrics.

---

### 🎛️ 3.3 Hyperparameter Optimization
**Tool**: **GridSearchCV** was utilized for systematic hyperparameter tuning to maximize model performance.

#### 1) **Model Structure Parameters**
- **`max_depth` (Tree Depth)**:
  - **Initial**: `-1` (unlimited depth)
  - **Optimal**: `7`
  - **Rationale**: Balances performance and prevents overfitting.
  
- **`num_leaves` (Number of Leaf Nodes)**:
  - **Rule**: Should follow \( 2^{\text{max_depth}} \)
  - **For `max_depth=7`**: \( 2^7 = 128 \)
  - **Optimal**: `45` (determined via GridSearch)

#### 2) **Sampling Parameters**
- **`subsample`**:
  - **Definition**: Fraction of data used per tree.
  - **Optimal**: `0.1`
  
- **`colsample_bytree`**:
  - **Definition**: Fraction of features used per tree.
  - **Optimal**: `0.4`

#### 3) **Regularization Parameters**
- **Purpose**: Prevent overfitting.
- **Parameters Tuned** (range: [0, 1] with step 0.1):
  - **`reg_alpha`** (L1 Regularization): `0.7`
  - **`reg_lambda`** (L2 Regularization): `0.1`
  - **`min_gain_to_split`** (Minimum Gain for Splitting): `0`

#### 4) **Leaf Node Parameters**
- **`min_data_in_leaf`**:
  - **Definition**: Minimum samples per leaf node.
  - **Explored Range**: [100, 1000] (step: 100)
  - **Optimal**: `700`

#### 5) **Additional Adjustments**
- **`bagging_freq`**:
  - **Definition**: Frequency of bagging.
  - **Optimal**: `0` (disables bagging)
  
- **`n_estimators`**:
  - **Definition**: Number of trees.
  - **Explored Range**: [1, 2000] (step: 100)
  - **Optimal**: `1200`
  
- **`max_bin`**:
  - **Definition**: Number of bins for feature discretization.
  - **Explored Range**: [100, 1000]
  - **Optimal**: `360`
  - **Effect**: Enhances accuracy, slightly slower training, prevents overfitting.

---

### 🌟 Final Optimized Hyperparameters
- `max_depth`: **7**
- `num_leaves`: **45**
- `subsample`: **0.1**
- `colsample_bytree`: **0.4**
- `reg_alpha`: **0.7**
- `reg_lambda`: **0.1**
- `min_data_in_leaf`: **700**
- `n_estimators`: **1200**
- `max_bin`: **360**

*These parameters significantly boosted model performance. Detailed evaluation results are available below.*

---

## 📈 4. Model Performance Evaluation

### 🏅 4.1 Results
- **Validation Data**:
  - **F1 Score**: `0.771309`
  - **AUC Score**: `0.871433`
  
- **Kaggle Data**:
  - **Public Score**: `0.85732`
  - **Private Score**: `0.87279`

---

## 📚 5. References
- [psystat Blog](https://psystat.tistory.com/131)
- [potato Blog](https://potato-potahto.tistory.com/entry/Light-GBM-%EC%84%A4%EB%AA%85%ED%8A%B9%EC%A7%95%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EC%84%A4%EC%B9%98-%EC%82%AC%EC%9A%A9%EB%B0%A9%EB%B2%95#google_vignette)
- [greatjoy Blog](https://greatjoy.tistory.com/72)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)

