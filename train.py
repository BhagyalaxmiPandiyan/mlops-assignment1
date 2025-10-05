"""Train a DecisionTreeRegressor on the Boston dataset and print MSE."""
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from misc import load_data, preprocess, get_kfold


def main():
    # Load data and preprocess once (we'll cross-validate on full data)
    df = load_data()
    X = df.drop(columns=['MEDV'])
    y = df['MEDV']
    X_t, _, _ = preprocess(X, X, standardize=True)

    model = DecisionTreeRegressor(random_state=42)

    # Use negative MSE scoring; cross_val_score returns array of scores
    cv = get_kfold(n_splits=5, random_state=42)
    scores = cross_val_score(model, X_t, y, cv=cv, scoring='neg_mean_squared_error')
    mses = -scores
    mean_mse = np.mean(mses)
    std_mse = np.std(mses)
    print(f"DecisionTreeRegressor 5-fold CV MSE: {mean_mse:.4f} ± {std_mse:.4f}")


if __name__ == '__main__':
    main()
