
# Boston Housing Price Prediction

This repository contains code to train two regression models on the Boston Housing dataset: a DecisionTreeRegressor and a KernelRidge model. The project follows the assignment instructions to keep reusable utilities in `misc.py`.

#Project Structure:
.
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── misc.py            # Utility functions for ML workflow
├── train.py           # DecisionTreeRegressor model
└── train2.py          # KernelRidge model


Dependencies
 - Python 3.8+
 - See `requirements.txt`

Install

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Run

```powershell
python train.py
python train2.py
```

This will:

Load the Boston Housing dataset
Preprocess and scale the data
Train a KernelRidge model
Evaluate and display the Mean Squared Error (MSE) on the test set

Models
1. DecisionTreeRegressor
A decision tree regression model that doesn't require feature scaling.

2. KernelRidge
A kernel ridge regression model that benefits from feature scaling.

Project Workflow
The project follows a modular approach with utility functions in misc.py:

load_data(): Loads the Boston Housing dataset
preprocess_data(): Splits data into train/test sets and optionally scales features
train_model(): Trains a given model
evaluate_model(): Evaluates the model and calculates MSE
display_results(): Displays evaluation result
