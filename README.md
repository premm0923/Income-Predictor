# 💰 Income Prediction Project 📈

A machine learning pipeline built with Python and Scikit-learn to predict whether an individual's annual income exceeds \$50,000 based on census data. This tool automates the process from data cleaning and feature engineering to model training, evaluation, and prediction.

---

## ✨ Features

- **🧹 Data Preprocessing:** Cleans the dataset by handling missing values and stripping whitespace.
- **📊 Exploratory Data Analysis (EDA):** Generates and displays visualizations for data distributions and feature correlations.
- **🛠️ Feature Engineering:** Creates a new feature, `capital_movement`, to improve model accuracy.
- **🤖 Model Training & Tuning:** Trains and tunes three different classification models:
  - Random Forest
  - Gradient Boosting
  - XGBoost
- **📈 Model Evaluation:** Compares models using the ROC AUC score and selects the best one for the task.
- **💾 Model Persistence:** Saves the best-trained model to a file (`.joblib`) for easy reuse.
- **💡 Prediction:** Loads the saved model to make predictions on new, unseen data points.

---

## ⚙️ How It Works

The application employs a standard machine learning pipeline to process the data and make predictions.

1. **Data Loading & Cleaning:**  
   The `adult 3.csv` dataset is loaded, and missing values (marked as `?`) are replaced with the mode of their respective columns.

2. **Preprocessing Pipeline:**  
   A `ColumnTransformer` is used to apply different transformations to different types of data.
   - **Numerical Features:** Standardized using `StandardScaler`.
   - **Categorical Features:** Converted into numerical format using `OneHotEncoder`.

3. **Model Training:**  
   Three powerful ensemble models are trained on the preprocessed data.

4. **Hyperparameter Tuning:**  
   `RandomizedSearchCV` is used to efficiently search for the best hyperparameters for each model, optimizing for the `roc_auc` score.

5. **Selection & Evaluation:**  
   The model with the highest ROC AUC score on the test set is selected as the best model. A detailed classification report is printed.

6. **Saving & Predicting:**  
   The final, optimized model is saved to `best_salary_predictor.joblib`. The script then demonstrates how to load this model and predict the income bracket for a new individual.

---

## 📁 Project Structure

```
.
├── income_predictor.py           
├── adult 3.csv                   
├── best_salary_predictor.joblib  
└── README.md                     
```

---

## ✅ Requirements

- Python 3.x (3.8+ recommended)
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- XGBoost
- Joblib

Install all dependencies with a single command:

```sh
pip install pandas numpy matplotlib seaborn joblib xgboost scikit-learn
```

---

## 🚀 How to Run

1. **Clone or Download** this repository.
2. **Install Dependencies** as shown above.
3. **Run the Application:**

   ```sh
   python income_predictor.py
   ```

   The script will run the entire pipeline and print the results to the console.

---

## ⚠️ Notes & Limitations

- The dataset `adult 3.csv` must be present in the same directory as the script for it to run correctly.
- This is a command-line application. The EDA step will generate and display plots during execution.
- The model's performance is dependent on the quality and characteristics of the training data.

---

## 💡 Example Prediction Output

After the model is trained, the script shows an example prediction on a new data point:

```
--- Loading model and making a new prediction ---
Prediction: The individual's income is likely >50K.
Confidence: 85.12%
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Credits

This project uses the "Adult" dataset, which is publicly available from the UCI Machine Learning Repository.

- **Source:** UCI Machine Learning Repository: [Adult Data Set](http://archive.ics.uci.edu/ml)
- **Citation:** Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
