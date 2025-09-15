import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def perform_eda(df):
    print("--- Starting Exploratory Data Analysis (EDA) ---")
    if df.empty:
        print("DataFrame is empty. Skipping EDA.")
        return
    print("\nDataset Info:")
    df.info()
    print("\nSummary Statistics for Numerical Features:")
    print(df.describe())
    print("\nVisualizing Data Distributions:")
    plt.figure(figsize=(12, 6))
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.show()
    plt.figure(figsize=(12, 6))
    sns.countplot(y='education', data=df, order=df['education'].value_counts().index)
    plt.title('Education Level Distribution')
    plt.show()
    corr_matrix = df.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()
    print("--- EDA Finished ---\n")


def engineer_features(df):
    df_engineered = df.copy()
    df_engineered['capital_movement'] = df_engineered['capital-gain'] - df_engineered['capital-loss']
    df_engineered = df_engineered.drop(['capital-gain', 'capital-loss'], axis=1)
    return df_engineered


def train_and_evaluate_models(x_train, y_train, x_test, y_test, preprocessor):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    param_grids = {
        'Random Forest': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 20, 30, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 7],
            'classifier__colsample_bytree': [0.7, 0.8, 1.0]
        }
    }

    best_model = None
    best_model_name = ""
    best_auc = 0.0

    for name, model in models.items():
        print(f"--- Training and Tuning {name} ---")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        random_search = RandomizedSearchCV(pipeline, param_grids[name], n_iter=10, cv=3, n_jobs=-1, scoring='roc_auc',
                                           random_state=42)
        random_search.fit(x_train, y_train)

        y_pred_proba = random_search.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

        print(f"\nBest Parameters for {name}: {random_search.best_params_}")
        print(f"AUC Score for {name}: {auc_score:.4f}\n")

        if auc_score > best_auc:
            best_auc = auc_score
            best_model = random_search.best_estimator_
            best_model_name = name

    print(f"\n--- Best Performing Model: {best_model_name} with AUC Score: {best_auc:.4f} ---")
    y_pred = best_model.predict(x_test)
    print("\nClassification Report for the Best Model:")
    print(classification_report(y_test, y_pred))
    return best_model


def save_model(model, filename):
    print(f"\nSaving the best model to {filename}...")
    joblib.dump(model, filename)
    print("Model saved successfully.")


def load_and_predict(data, filename="best_salary_predictor.joblib"):
    print(f"\n--- Loading model and making a new prediction ---")
    loaded_model = joblib.load(filename)

    if not isinstance(data, pd.DataFrame):
        data_df = pd.DataFrame([data])
    else:
        data_df = data

    data_engineered = engineer_features(data_df)
    prediction_coded = loaded_model.predict(data_engineered)
    prediction_proba = loaded_model.predict_proba(data_engineered)
    prediction = '<=50K' if prediction_coded[0] == 0 else '>50K'
    confidence = prediction_proba[0][prediction_coded[0]] * 100

    print(f"Prediction: The individual's income is likely {prediction}.")
    print(f"Confidence: {confidence:.2f}%")
    return prediction, confidence


def main():
    try:
        df = pd.read_csv('adult 3.csv')
    except FileNotFoundError:
        print("Error: The file 'adult 3.csv' was not found.")
        return

    df.replace('?', np.nan, inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    for col in ['workclass', 'occupation', 'native-country']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    if 'income' not in df.columns:
        print("Error: 'income' column not found in the dataset.")
        return

    perform_eda(df.copy())
    df_processed = engineer_features(df)

    x = df_processed.drop('income', axis=1)
    y = df_processed['income'].apply(lambda x: 1 if x == '>50K' else 0)

    numerical_features = x.select_dtypes(include=np.number).columns
    categorical_features = x.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=0.01), categorical_features)
        ])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    best_model = train_and_evaluate_models(x_train, y_train, x_test, y_test, preprocessor)

    if best_model:
        save_model(best_model, "best_salary_predictor.joblib")

        new_individual_data = {
            'age': 35, 'workclass': 'Private', 'fnlwgt': 200000,
            'education': 'Bachelors', 'educational-num': 13,
            'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial',
            'relationship': 'Husband', 'race': 'White', 'gender': 'Male',
            'capital-gain': 5000, 'capital-loss': 0, 'hours-per-week': 45,
            'native-country': 'United-States'
        }
        load_and_predict(new_individual_data)


if __name__ == "__main__":
    main()