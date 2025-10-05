# Assignment 1 — Report (template)

Repository: add your GitHub URL here

## Models and performance (5-fold CV)

| Model | Mean MSE | Std MSE |
|---|---:|---:|
| DecisionTreeRegressor | 19.9592 | 6.9443 |
| KernelRidge (tuned) | 13.3184 | 4.1053 |

Replace the numbers above with the outputs you get locally or from CI.

## How to reproduce

1. Clone repo and create branches (example):

```powershell
git clone <GITHUB_REPO_URL>
cd <repo>
git checkout -b dtree
git push -u origin dtree
git checkout main
git checkout -b kernelridge
git push -u origin kernelridge
```

2. Run locally:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
# run
python train.py
python train2.py
```

## GitHub Actions

- The workflow `.github/workflows/push.yml` runs on pushes to `main`, `dtree`, and `kernelridge`.
- After pushing, go to the Actions tab on GitHub, open the workflow run, and take a screenshot of the log that shows the MSE outputs.

## Screenshots to include

- Branch listing showing `main`, `dtree`, and `kernelridge`.
- GitHub Actions run log showing both model outputs.

## Notes

- KernelRidge was tuned with GridSearchCV over alpha, kernel and gamma. You can extend the grid for a better search.
- The `misc.py` has helper functions and a reproducible `get_kfold` function.
