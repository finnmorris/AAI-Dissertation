# AAI-Dissertation — Expected Goals (xG) Model

An expected goals (xG) model for football built on **StatsBomb open event data**, predicting the probability that a shot results in a goal from spatial, contextual, and freeze-frame features.

Three models are trained and compared: **XGBoost** (primary), **Ridge logistic regression** (calibrated baseline), and **Lasso logistic regression** (feature selection via sparsity). Champions League 2018/19 is held out entirely as a test set across all models.

## Project structure

```
src/
  features/         # Feature engineering — create_xg_features()
  preprocessing/     # Data cleaning and pipeline assembly
  data/               # StatsBomb data loading
  models/             # Model training
  evaluation/         # Calibration, log-loss, Brier score, ROC/AUC
  interpretability/   # SHAP, feature importance
notebooks/
  01_eda.ipynb            # Exploratory analysis
  02_xgboost_model.ipynb  # XGBoost model
  03_ridge_model.ipynb    # Ridge logistic regression model
  04_lasso_model.ipynb    # Lasso logistic regression model
outputs/    # Saved models, feature reference, figures
data/       # Local-only StatsBomb open data (not committed)
```

## Setup

```bash
conda create -n aai-dissertation python=3.11
conda activate aai-dissertation
pip install -r requirements.txt
```

## Running the system

1. **Get the data.** StatsBomb open data is not committed to this repo. Either clone [statsbomb/open-data](https://github.com/statsbomb/open-data) into `data/`, or fetch it at runtime via the `statsbombpy` package (already used in the notebooks).
2. **Run the notebooks in order** from the project root, via Jupyter or your IDE:
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```
   - `01_eda.ipynb` — loads shot events, builds the feature set, explores the data
   - `02_xgboost_model.ipynb` — trains/evaluates the XGBoost model
   - `03_ridge_model.ipynb` — trains/evaluates the Ridge baseline
   - `04_lasso_model.ipynb` — trains/evaluates the Lasso model
3. **Use the feature pipeline directly** in your own code:
   ```python
   from src.features import create_xg_features
   xg_df = create_xg_features(events_df)
   ```

Trained models and figures are written to `outputs/`.
