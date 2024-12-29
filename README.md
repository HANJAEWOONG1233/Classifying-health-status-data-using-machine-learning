#  Building and Optimizing a Health Status Classification Model Using Healthcare Data 

## 🚀 1. Introduction

### 🎯 1.1 Objective of the Study
- **Build an optimal machine learning model** for classifying patients' health status (`health_status`) using healthcare data.
- **Target Variable**:
  - `0` (**Normal**): Healthy condition
  - `1` (**Abnormal**): Unhealthy condition

---

## 🛠️ 2. Data Preprocessing and Feature Selection

### 📊 2.1 Dataset Overview and Missing Value Check
- **Dataset Overview**:
  - **Total Data Size**: 140,000 samples
  - **Number of Variables**: 26 (continuous and categorical variables)

| **Variable Name**    | **Description**                     |
|----------------------|-------------------------------------|
| `index`              | Unique index                        |
| `name`               | Patient name                        |
| `birth`              | Year of birth                       |
| `bp_high`, `bp_low`  | Blood pressure values               |
| `chol_good`, `chol_bad` | Good/Bad cholesterol levels     |
| `iron_lvl`, `creatinine` | Iron/Creatinine levels          |
| `urine_marker`      | Urine marker values                 |
| `sugar_lvl`, `lipid_lvl`, `fat_content` | Sugar, lipid, and fat levels |
| `oral_issues`       | Oral health issues                  |
| `stature_inch`, `mass_kg`, `midsection_inch` | Height, weight, and waist size |
| `immune_index`      | Immunity index                      |
| `enzyme_1`, `enzyme_2`, `enzyme_3` | Enzyme levels           |
| `vision_l`, `vision_r` | Left/Right vision               |
| `audio_l`, `audio_r` | Left/Right hearing test results     |
| `health_status`     | Health status (target variable)     |

- **Missing Value Check**:
  - **Result**: No missing values were found using `train_df.isna().sum()`.

---

### 🔍 2.2 Feature Importance Evaluation Using Mutual Information
- **Objective**: Evaluate feature importance to identify and remove less relevant features.
  - **Evaluation Criteria**: F1 Score and AUC Score using the LGBM model.
  - **Removal Condition**: Remove features if both scores improve.
  - **Retention Condition**: Retain features if scores remain constant or degrade.

---

### ❌ 2.3 Removed Features
- **Continuous Variables**:
  - `immune_index`: Removal improved F1 Score to `0.72401` and AUC Score to `0.844841`.
  - `enzyme_1`: Removal improved F1 Score to `0.723322` and AUC Score to `0.844418`.
- **Categorical Variables**:
  - `urine_marker`, `audio_l`, `audio_r`, `name`, `index`

---

### ➕ 2.4 Generated Features and Validation
- **Generated Features**:
  - **`age`**: Calculated using year of birth.  
    `age = 2024 – birth`
  - **`BMI`**: Calculated using height and weight.  
    `BMI = mass_kg / (stature_inch × 0.0254)^2`
- **Validation**:
  - **Adding `age`** improved F1 Score to `0.72686` and AUC Score to `0.847056`.
  - **Adding `BMI`** improved F1 Score to `0.730212` and AUC Score to `0.847312`.

---

## 🤖 3. Model Building and Optimization

### ⚡ 3.1 Overview of LightGBM Model
**LightGBM** is a high-performance machine learning model based on Gradient Boosting. For this project, the **DART (Dropout Additive Regression Trees)** boosting method was utilized.

- **DART**:
  - **Applies** the dropout technique from neural networks to tree-based models to address over-specialization problems.
  - **Working Principle**: Instead of dropping individual features, entire trees are omitted at the tree level.
  - **Advantages**:
    - Prevents overfitting.
    - Improves generalization performance.

---

### 🔀 3.2 Data Splitting (Using Stratified Sampling)
The dataset was split into **90% training data** and **10% validation data**, with **Stratified Sampling** applied to maintain the proportion of the target variable (`health_status`).

- **Stratified Sampling**:
  - **Ensures** the data distribution ratio is preserved during sampling.
  - **Example**: If the population has a class distribution of 54% and 46%, the sample will maintain the same ratio.
  - **Benefits**:
    - Alleviates data imbalance issues.
    - Enables stable model performance evaluation.

- **Rationale for Use**:
  - Random splitting may cause class imbalances.
  - Stratified sampling preserves data distribution and ensures consistent evaluation metrics.

### 🎛️ 3.3 Hyperparameter Optimization
To maximize the model's performance, **GridSearchCV** was used to systematically tune the hyperparameters.

#### 1) Model Structure Parameters
- **`max_depth` (Tree Depth)**:
  - Initially set to `-1` (unlimited depth) and used as the baseline performance.
  - GridSearch and manual exploration revealed that setting `max_depth=7` achieved similar performance while reducing overfitting and training time.
  - **Rationale**: Appropriate tree depth ensures balanced performance and prevents overfitting.
  
- **`num_leaves` (Number of Leaf Nodes)**:
  - Theoretical rule: `num_leaves` should follow \( 2^{\text{max_depth}} \).
  - For `max_depth=7`, \( 2^7 = 128 \). The optimal value was determined to be **`num_leaves=45`** through GridSearch.

#### 2) Sampling Parameters
- **`subsample`**:
  - Fraction of data used for training each tree.  
  - **Optimal Value**: `0.1`
- **`colsample_bytree`**:
  - Fraction of features used for building each tree.  
  - **Optimal Value**: `0.4`

#### 3) Regularization Parameters
- To prevent overfitting, the following parameters were tuned within the range [0, 1] with a step of 0.1:
  - **`reg_alpha`** (L1 Regularization): **`0.7`**
  - **`reg_lambda`** (L2 Regularization): **`0.1`**
  - **`min_gain_to_split`** (Minimum Gain for Splitting): **`0`**

#### 4) Leaf Node Parameters
- **`min_data_in_leaf`**:
  - Minimum number of data points in a leaf node to prevent overfitting.
  - Explored values in the range [100, 1000] with steps of 100.
  - **Optimal Value**: `700`

#### 5) Additional Adjustments
- **`bagging_freq`**:
  - Controls the frequency of bagging. Setting it to `0` disables bagging.
  - **Optimal Value**: `0`
- **`n_estimators`**:
  - Number of trees in the model.
  - Explored values between [1, 2000] with steps of 100.
  - **Optimal Value**: `1200`
- **`max_bin`**:
  - Number of bins for discretizing features.
  - Explored values between [100, 1000].
  - **Optimal Value**: `360`
  - **Effect**: Improves accuracy at the cost of slightly slower training, while preventing overfitting.

---

The final optimal hyperparameters significantly improved the model's performance. For further details, refer to the model evaluation results.

- **Optimized Parameters**:
  - `max_depth`: **7**
  - `num_leaves`: **45**
  - `subsample`: **0.1**
  - `colsample_bytree`: **0.4**
  - `reg_alpha`: **0.7**
  - `reg_lambda`: **0.1**
  - `min_data_in_leaf`: **700**
  - `n_estimators`: **1200**
  - `max_bin`: **360**

---

## 📈 4. Model Performance Evaluation

### 🏅 4.1 Results
- **Validation Data**:
  - **F1 Score**: `0.771309`
  - **AUC Score**: `0.871433`
- **Kaggle Data**:
  - **Public Data**: `0.85732`
  - **Private Data**: `0.87279`

---

## 📚 5. References
- [psystat Blog](https://psystat.tistory.com/131)
- [potato Blog](https://potato-potahto.tistory.com/entry/Light-GBM-%EC%84%A4%EB%AA%85%ED%8A%B9%EC%A7%95%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EC%84%A4%EC%B9%98-%EC%82%AC%EC%9A%A9%EB%B0%A9%EB%B2%95#google_vignette)
- [greatjoy Blog](https://greatjoy.tistory.com/72)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)

