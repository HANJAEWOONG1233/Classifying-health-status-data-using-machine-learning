# Building and Optimizing a Health Status Classification Model Using Healthcare Data

## 1. Introduction

### 1.1 Objective of the Study
- Build an optimal machine learning model for classifying patients' health status (`health_status`) using healthcare data.
- **Target variable**:
  - 0 (`normal`): Healthy condition
  - 1 (`abnormal`): Unhealthy condition

---

## 2. Data Preprocessing and Feature Selection

### 2.1 Dataset Overview and Missing Value Check
- **Dataset Overview**:
  - Total data size: 140,000 samples
  - Number of variables: 26 (continuous and categorical variables)

| Variable Name      | Description                   |
|--------------------|-------------------------------|
| `index`           | Unique index                  |
| `name`            | Patient name                  |
| `birth`           | Year of birth                 |
| `bp_high`, `bp_low` | Blood pressure values         |
| `chol_good`, `chol_bad` | Good/Bad cholesterol levels |
| `iron_lvl`, `creatinine` | Iron/Creatinine levels      |
| `urine_marker`    | Urine marker values           |
| `sugar_lvl`, `lipid_lvl`, `fat_content` | Sugar, lipid, and fat levels |
| `oral_issues`     | Oral health issues            |
| `stature_inch`, `mass_kg`, `midsection_inch` | Height, weight, and waist size |
| `immune_index`    | Immunity index                |
| `enzyme_1`, `enzyme_2`, `enzyme_3` | Enzyme levels          |
| `vision_l`, `vision_r` | Left/Right vision           |
| `audio_l`, `audio_r` | Left/Right hearing test results |
| `health_status`   | Health status (target variable)|

- **Missing Value Check**:
  - No missing values were found using `train_df.isna().sum()`.

---

### 2.2 Feature Importance Evaluation Using Mutual Information
- Evaluate feature importance to identify and remove less relevant features:
  - **Evaluation criteria**: F1 Score and AUC Score using the LGBM model.
  - **Removal condition**: Remove features if both scores improve.
  - **Retention condition**: Retain features if scores remain constant or degrade.

---

### 2.3 Removed Features
- **Continuous Variables**:
  - `immune_index`: Removal improved F1 Score to 0.72401 and AUC Score to 0.844841.
  - `enzyme_1`: Removal improved F1 Score to 0.723322 and AUC Score to 0.844418.
- **Categorical Variables**:
  - `urine_marker`, `audio_l`, `audio_r`, `name`, `index`

---

### 2.4 Generated Features and Validation
- **Generated Features**:
  - `age`: Calculated using year of birth.  
    `age = 2024 – birth`
  - `BMI`: Calculated using height and weight.  
    `BMI = mass_kg / (stature_inch × 0.0254)^2`
- **Validation**:
  - Adding `age` improved F1 Score to 0.72686 and AUC Score to 0.847056.
  - Adding `BMI` improved F1 Score to 0.730212 and AUC Score to 0.847312.

---

## 3. Model Building and Optimization

### 3.1 Overview of LightGBM Model
- **Method**: DART (Dropout Additive Regression Trees)
  - **Advantages**: Prevents overfitting and improves generalization performance.

---

### 3.2 Data Splitting
- **Split Ratio**: 90% training data, 10% validation data
- **Method**: Stratified sampling to maintain class proportions.

---

### 3.3 Hyperparameter Optimization
- **Optimized Parameters**:
  - `max_depth`: 7
  - `num_leaves`: 45
  - `subsample`: 0.1
  - `colsample_bytree`: 0.4
  - `reg_alpha`: 0.7
  - `reg_lambda`: 0.1
  - `min_data_in_leaf`: 700
  - `n_estimators`: 1200
  - `max_bin`: 360

---

## 4. Model Performance Evaluation

### 4.1 Results
- **Validation Data**:
  - F1 Score: 0.771309
  - AUC Score: 0.871433
- **Kaggle Data**:
  - Public Data: 0.85732
  - Private Data: 0.87279

---

## 5. References
- [psystat Blog](https://psystat.tistory.com/131)
- [potato Blog](https://potato-potahto.tistory.com/entry/Light-GBM-%EC%84%A4%EB%AA%85%ED%8A%B9%EC%A7%95%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EC%84%A4%EC%B9%98-%EC%82%AC%EC%9A%A9%EB%B0%A9%EB%B2%95#google_vignette)
- [greatjoy Blog](https://greatjoy.tistory.com/72)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)
