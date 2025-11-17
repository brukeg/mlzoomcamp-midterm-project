import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

DATA_PATH = "salary-job-data.csv"
MODEL_FILE = "model_decision_tree.bin"

def rmse(y_true, y_pred):
    """Return root mean squared error for arrays of true and predicted values."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and string values, then drop rows with missing values.

    - Columns lowercased
    - Spaces to underscores (camel_case)
    - String columns lowercased
    - Rows with any NaNs removed
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.dropna()
    strings = list(df.dtypes[df.dtypes == 'object'].index)
    for col in strings:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    return df

if __name__ == "__main__":
    np.random.seed(1)

    # Load and clean
    df = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df)

    # Split: 60% train, 20% val, 20% test overall (via 80/20 then 75/25)
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    y_train = df_train.salary.values
    y_val = df_val.salary.values
    y_test = df_test.salary.values

    # Remove salary from features
    for part in (df_train, df_val, df_test):
        del part["salary"]

    # Vectorize features
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train.to_dict(orient="records"))
    X_val = dv.transform(df_val.to_dict(orient="records"))
    X_test = dv.transform(df_test.to_dict(orient="records"))

    # Train model
    model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, random_state=1)
    model.fit(X_train, y_train)

    # Report metrics
    print(f"Train RMSE: {rmse(y_train, model.predict(X_train)) :,.2f}")
    print(f"Validation RMSE: {rmse(y_val,   model.predict(X_val))   :,.2f}")
    print(f"Test RMSE: {rmse(y_test,       model.predict(X_test))  :,.2f}")

    # Save model
    with open(MODEL_FILE, "wb") as f_out:
        pickle.dump((dv, model), f_out)

    print(f"Saved model to {MODEL_FILE}")
