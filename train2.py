"""Train a KernelRidge model on the Boston dataset and print MSE."""
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import numpy as np
from misc import load_data, preprocess, get_kfold


def main():
    df = load_data()
    X = df.drop(columns=['MEDV'])
    y = df['MEDV']
    X_t, _, _ = preprocess(X, X, standardize=True)

    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.1, 1.0, 10.0]
    }
    kr = KernelRidge()
    cv = get_kfold(n_splits=5, random_state=42)
    gs = GridSearchCV(kr, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    gs.fit(X_t, y)
    best = gs.best_estimator_
    best_score = -gs.best_score_
    # get CV results for best params across folds
    print(f"KernelRidge best params: {gs.best_params_}")
    # To compute std across folds, extract results for the best param index
    best_index = gs.best_index_
    mean_mse = -gs.cv_results_['mean_test_score'][best_index]
    std_mse = gs.cv_results_['std_test_score'][best_index]
    print(f"KernelRidge 5-fold CV MSE (best): {mean_mse:.4f} ± {std_mse:.4f}")


if __name__ == '__main__':
    main()
