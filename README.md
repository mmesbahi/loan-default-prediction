
# German Credit Risk Prediction Project

## Table of Contents
* [Project Goal](#project-goal)
* [Dataset](#dataset)
* [Project Workflow](#project-workflow)
* [Model Performance & Results](#model-performance--results)
* [Technologies Used](#technologies-used)
* [How to Run This Project](#how-to-run-this-project)

---

## Project Goal

The objective of this end-to-end data science project is to build and evaluate several machine learning models to predict credit default risk using the German Credit Data dataset.

The primary business goal is **not** to maximize overall accuracy, but to **minimize False Negatives (FN)**. A False Negative—classifying a high-risk client as "Good"—is the most costly error for a bank. This project analyzes the trade-offs required to build a model that is both effective and aligned with this business priority.

---

## Dataset

This project uses the "German Credit Data" dataset, which contains 1,000 entries with 10 features, including:
* `Age`
* `Sex`
* `Job`
* `Housing`
* `Saving accounts`
* `Checking account`
* `Credit amount`
* `Duration`
* `Purpose`

The target variable is `Risk`, which was encoded to `bad` (Bad Risk) and `good` (Good Risk).

---

## Project Workflow

The project followed a structured data science pipeline:

1.  **Exploratory Data Analysis (EDA):**
    * Analyzed all features to understand their distributions and relationships with `Risk`.
    * Identified significant outliers in the `Credit amount` column.
    * Discovered a critical **class imbalance**: the dataset is ~70% "Good" (`0`) and 30% "Bad" (`1`).

2.  **Data Preprocessing & Cleaning:**
    * Dropped the redundant `Unnamed: 0` column.
    * Handled ~577 missing values in `Saving accounts` and `Checking account` by imputing a new `'Unknown'` category, preserving this information.

3.  **Feature Engineering:**
    * Encoded binary categorical features (`Sex`) to `0`/`1`.
    * **One-Hot Encoded** all multi-categorical features (`Job`, `Housing`, `Purpose`, etc.) using `pd.get_dummies(drop_first=True)` to prevent multicollinearity.
    * **Scaled** all features using `StandardScaler` to make the data robust to outliers, which was identified as a key issue for linear models.

4.  **Model Training & Evaluation:**
    * Split the data into training and test sets (80/20).
    * Trained and evaluated five different models to find the best solution for the class imbalance problem.

---

## Model Performance & Results

The key challenge was the class imbalance. We compared models based on their ability to minimize our critical error: **False Negatives (FN)**.

| Model | Key Strategy | FN (Risk Missed) | FP (Good Rejected) | Conclusion |
| :--- | :--- | :---: | :---: | :--- |
| 1. Logistic Regression | (Baseline) | 35 | 13 | **Unacceptable.** Fails to identify risk. |
| 2. **Logistic Regression** | **`class_weight='balanced'`** | **22** | **42** | **Best Model.** Finds the best trade-off. |
| 3. Random Forest | (Baseline) | 37 | 12 | **Failed.** Worse than the baseline. |
| 4. Random Forest | `class_weight='balanced'` | 39 | 9 | **Failed.** Parameter was ineffective. |
| 5. SMOTE + Random Forest | (Data Resampling) | 34 | 22 | **Failed.** Model overfit to synthetic data. |

### Final Decision

The **Model 2: Logistic Regression (Balanced)** was selected as the final, most robust solution.

* `[[TN=99, FP=42], [FN=22, TP=37]]`
* **Recall (Risk `1`): 63%**
* **Precision (Risk `1`): 47%**

This model achieved the **lowest number of False Negatives (22)**, successfully identifying 63% of all high-risk clients. It demonstrates the best real-world compromise, accepting a higher number of False Positives (42) to successfully mitigate the most expensive error (FN). This project proves that a simple, well-tuned model that directly addresses the core data problem (imbalance) is superior to a more complex model that does not.

---

## Technologies Used
* Python (3.12.7)
* Pandas (for data manipulation)
* Scikit-learn (for modeling, scaling, and metrics)
* Imbalanced-learn (for SMOTE)
* Seaborn & Matplotlib (for visualization)
* Jupyter Notebook

---

## How to Run This Project

1.  Clone the repository:
    ```sh
    git clone [https://github.com/](https://github.com/)[mmesbahi/loan-default-prediction.git
    ```
2.  Navigate to the project directory:
    ```sh
    cd loan-default-prediction
    ```
3.  Create and activate a virtual environment:
    ```sh
    python -m venv venv
    .\venv\Scripts\activate
    ```
4.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
5.  Launch the Jupyter Notebook:
    ```sh
    jupyter notebook
    ```
