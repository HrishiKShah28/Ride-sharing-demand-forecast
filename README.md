# Bike Sharing Demand Forecasting Project 🚲

## 📝 Description
This project focuses on predicting the hourly demand for bike rentals using historical data. It utilizes the dataset from a past Kaggle competition ("Bike Sharing Demand") to build a regression model that forecasts the `count` of bike rentals based on temporal and environmental factors.

---

## 💾 Dataset
The data used is the **Bike Sharing Demand** dataset from Kaggle. It includes information such as:
* Timestamp (`datetime`)
* Weather conditions (`weather`, `temp`, `atemp`, `humidity`, `windspeed`)
* Seasonal and calendar information (`season`, `holiday`, `workingday`)
* Rental counts (`casual`, `registered`, `count` - the target)

---

## ⚙️ Workflow & Key Steps
1.  **Exploratory Data Analysis (EDA):** Analyzed feature distributions and relationships with the target variable (`count`). Identified significant right-skewness in the target.
2.  **Feature Engineering:**
    * Applied a **log transformation** (`np.log1p`) to the `count` variable to create the `log_count` target for better model performance.
    * Extracted temporal features (hour, day of week, month, year) from the `datetime` column.
    * Engineered **cyclical features** using sine and cosine transformations for `hour` and `month` to capture cyclical patterns.
    * Created interaction features (e.g., `temp * humidity`).
3.  **Preprocessing:**
    * Set up a `scikit-learn` `ColumnTransformer` to apply `StandardScaler` to numerical features and `OneHotEncoder` to categorical features (`season`, `weather`).
4.  **Modeling:**
    * Integrated the preprocessor into a `scikit-learn` `Pipeline`.
    * Trained an **XGBoost Regressor** (`XGBRegressor`) model within the pipeline to predict `log_count`.
5.  **Evaluation:**
    * Split the data using `train_test_split` (Note: `TimeSeriesSplit` is recommended for robust time-series validation).
    * Evaluated the model using R² and **Root Mean Squared Logarithmic Error (RMSLE)** on the inverse-transformed predictions (`np.expm1`).
6.  **Prediction:**
    * Applied the same feature engineering and the trained pipeline to the `test.csv` data.
    * Generated a `submission.csv` file with predictions in the required format.

---

## ✨ Techniques Used
* Log Transformation for skewed target variables
* Time-based Feature Extraction
* Cyclical Feature Engineering (Sine/Cosine)
* Interaction Features
* Scikit-learn Pipelines & Column Transformers
* XGBoost Regression
* RMSLE Evaluation Metric

---

## 🛠️ Technologies
* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn
* Jupyter Notebook

---

## 📊 Results
The final model achieved a Root Mean Squared Logarithmic Error (RMSLE) score of approximately **0.47** on the Kaggle test set (past competition).

---
