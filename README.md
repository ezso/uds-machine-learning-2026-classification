# UdS Machine Learning 2026 — Classification

Notebook-based experiments for predicting hourly bike-demand **`Demand_Category`** on the Seoul Bike Sharing dataset from weather, calendar, and bike-service features. The project compares logistic regression, tree ensembles, and neural networks across several cleaning and feature-engineering strategies.

## Data

The supplied `train.csv` contains **7,093 rows** with the classification target; `test.csv` contains **2,628 rows** without it. Features include date and hour, temperature, humidity, wind speed, visibility, dew point, solar radiation, rainfall, snowfall, season, holiday status, and functioning-day status. `Kaggle_ID` identifies rows in prediction files.

| Target value | Demand label |
| --- | --- |
| `0` | Low |
| `1` | Normal |
| `2` | High |

| File | Purpose |
| --- | --- |
| `train.csv`, `test.csv` | Original training and test data |
| `sample_submission.csv` | Example submission; its target column uses text labels |
| `train_clean.csv` | Cleaned data with anomalous rows and missing-value rows removed |
| `train_clean_nan.csv` | Cleaned data retaining missing values |
| `train_clean_imputed.csv` | Cleaned data with missing values imputed |
| `train_preserved_nan.csv` | Alternative data with anomalous readings replaced by missing values |
| `train_preserved_imputed.csv` | Imputed version of the preserved-data variant |
| `submissions/` | Previously saved submission files |
| `catboost_info/` | CatBoost training artifacts |
| `nn_run.log`, `tree_run.log` | Saved experiment logs |

The cleaning workflow inspects data quality, corrects season labels using dates, adds month, and compares removing anomalous rows with replacing individual readings by missing values. Weather thresholds are experiment assumptions that change the available training population.

The committed cleaned datasets contain 4,949 rows in `train_clean.csv`, 6,311 in the cleaned missing-value and imputed variants, and 6,455 in the preserved variants. Preserved variants retain rows after invalid-hour filtering; they do not retain every row of the original 7,093-row input.

## Notebooks

| Notebook | Contents |
| --- | --- |
| [data_cleaning_analysis.ipynb](data_cleaning_analysis.ipynb) | Exploratory analysis, anomaly handling, and generation of cleaned and imputed datasets |
| [logistic_regression.ipynb](logistic_regression.ipynb) | Logistic-regression baseline with imputation, scaling, and one-hot encoding |
| [tree_models.ipynb](tree_models.ipynb) | Random Forest, XGBoost, LightGBM, and CatBoost; cross-validation, classification diagnostics, and final LightGBM predictions |
| [nn_models.ipynb](nn_models.ipynb) | Basic tabular network, entity embeddings, and FT-Transformer with three output logits and cross-entropy loss |

The main comparison uses four datasets: raw, preserved/imputed, cleaned/imputed, and fully cleaned. Each passes through five cumulative experiment stages:

1. Baseline features.
2. Transformations of skewed weather variables.
3. Weather-category features.
4. Addition of weekday.
5. Cyclic time encodings.

Evaluation uses **five-fold stratified cross-validation** and **macro-F1**, which gives each class equal weight. Higher scores are better. The tree notebook additionally evaluates XGBoost and LightGBM directly on `train_clean_nan.csv` and `train_preserved_nan.csv` to compare native missing-value handling with imputation.

## Recorded results

The updated notebooks report the following best configurations from their saved experiments. These are validation results, not test-set leaderboard scores or newly rerun results.

| Model | Dataset and feature stage | Macro-F1 |
| --- | --- | --- |
| Logistic regression | `train_clean`, cyclic time | 0.7391 ± 0.0150 |
| FT-Transformer | `train_clean_imputed`, cyclic time | 0.8844 ± 0.0054 |
| LightGBM | `train_clean_nan`, cyclic time and native missing-value handling | 0.9028 |

The final submission workflow selects the LightGBM configuration. Its tree notebook also contains out-of-fold classification reports, a confusion matrix, and per-class metrics for inspecting mistakes beyond the aggregate score.

## Setup

Install Python and create an environment in the repository root. For Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn catboost xgboost lightgbm torch
.\.venv\Scripts\python.exe -m jupyter lab
```

These packages are inferred from notebook imports. There is no pinned dependency manifest, so this provides a starting environment rather than an exact reproduction of the original runs. The neural-network notebook selects CUDA, then Apple MPS when available, then CPU.

## Running the experiments

1. Start Jupyter from the repository root so relative CSV paths resolve correctly.
2. Run `data_cleaning_analysis.ipynb` to regenerate training variants, or use the committed CSV files.
3. Run `logistic_regression.ipynb` for a baseline, then `tree_models.ipynb` and/or `nn_models.ipynb` for model comparisons. Each model notebook contains its own experiment code; running the logistic notebook first is optional.
4. Execute cells in order within each notebook. Later diagnostics and prediction cells depend on earlier definitions and data.
5. In `nn_models.ipynb`, set `QUICK_MODE = True` for a smaller initial run. The full comparison covers four datasets, five stages, three architectures, and five folds.
6. Run the final cells of `tree_models.ipynb` to train the selected LightGBM configuration and export test predictions.

## Prediction output

The final tree-notebook workflow uses `train_clean_nan` with cyclic calendar features, fits LightGBM on the selected training data, and writes these files to the repository root:

- `classification_predictions_lightgbm_with_labels.csv`: row IDs, numeric predictions, and human-readable labels.
- `submission_classification_lightgbm.csv`: `Kaggle_ID` and numeric `Demand_Category` predictions (`0`, `1`, or `2`), sorted by ID.

**Submission format discrepancy:** the supplied `sample_submission.csv` uses labels such as `Low`, while the final notebook exports numeric class codes. Check the competition's required representation before uploading and map codes to `Low`, `Normal`, and `High` if text labels are required.

The final cells check row counts, class values, missing predictions, and matching feature columns. Also verify that test IDs appear exactly once. Existing outputs are stored in `submissions/`; the final export cells write to the root.

## Reproducibility notes

Restart the kernel before a fresh run. Neural-network preprocessing fits imputers and encoders within each training fold, but the pre-imputed CSV variants were generated before model cross-validation; their scores may benefit from information shared across folds. For strict model-selection estimates, fit all learned preprocessing within the training fold. Cleaning variants contain different numbers of rows, which also affects direct comparisons between their scores.

The final summary pivot in `logistic_regression.ipynb` uses `.str.split(" | ")`, which treats the separator as a regular expression and can produce duplicate pivot entries. Use `.str.split(" | ", regex=False)` in both split expressions before running that summary cell. Earlier result tables remain available. The READMEs document the current notebooks; notebook code has not been changed.
