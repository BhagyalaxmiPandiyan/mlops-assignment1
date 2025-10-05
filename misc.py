import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load Boston dataset from the CMU repository and return DataFrame."""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['MEDV'])
    y = df['MEDV']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess(X_train, X_test, standardize=True):
    """Optional standard scaling. Returns transformed arrays and the scaler."""
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train_t = scaler.fit_transform(X_train)
        X_test_t = scaler.transform(X_test)
    else:
        X_train_t = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_t = X_test.values if hasattr(X_test, 'values') else X_test
    return X_train_t, X_test_t, scaler


def get_kfold(n_splits=5, random_state=42, shuffle=True):
    """Return a sklearn.model_selection.KFold object for reproducible CV splits."""
    from sklearn.model_selection import KFold
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def train_model(model, X_train, y_train):
    """Fit model and return it."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return mean squared error of model predictions on test set."""
    from sklearn.metrics import mean_squared_error
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse
