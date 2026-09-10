# Boston Housing Price Prediction — Multivariable Regression

Predicting 1970s Boston home prices from property and neighborhood characteristics using multivariable linear regression, with a log-transformation step to improve model fit.

## Overview

This project builds a regression model to estimate a home's value based on features like room count, distance to employment centers, pollution levels, and neighborhood socioeconomic indicators. It walks through the full workflow: exploratory data analysis, model training, coefficient interpretation, residual diagnostics, and using data transformation to improve performance.

## Dataset

The classic Boston Housing dataset — 506 observations, 13 predictive features, with `PRICE` (median home value in $1000s) as the target. Features include `RM` (rooms per dwelling), `NOX` (pollution), `DIS` (distance to employment centers), `LSTAT` (% lower-status population), `PTRATIO` (pupil-teacher ratio), and `CHAS` (borders the Charles River).

## Tech Stack

- Python, pandas, NumPy
- Seaborn, Matplotlib, Plotly (visualization)
- scikit-learn (`LinearRegression`, `train_test_split`)

## What This Covers

1. **EDA** — distributions of key features, checks for missing values/duplicates, relationship exploration via joint plots
2. **Train/test split** — 80/20 split to evaluate out-of-sample performance
3. **Multivariable regression** — fitting a linear model across all 13 features
4. **Coefficient interpretation** — sense-checking coefficient signs against domain expectations
5. **Residual analysis** — checking for skew and systematic bias in prediction errors
6. **Log transformation** — applying `np.log()` to the target to correct right-skew and improve fit
7. **Model comparison** — evaluating both models on held-out test data
8. **Custom prediction** — estimating a price for a hypothetical property with specified characteristics

## Results

| Model | Train R² | Test R² |
|---|---|---|
| Linear regression (raw price) | 0.750 | 0.671 |
| Linear regression (log price) | 0.793 | 0.745 |

The log-transformed model fits better and generalizes better — residual skew dropped from 1.46 to 0.09 after the transformation.

**Example prediction:** for an 8-room home next to the river, in a low pupil-teacher-ratio area, with high pollution and low poverty in the surrounding area, the model estimates a price of **$25,792**.

## Key Takeaways

- Log-transforming a right-skewed target can meaningfully improve both fit and generalization for linear models.
- Residual diagnostics (skew, mean, and pattern) matter as much as R² for judging whether a linear model is actually appropriate.
- When predicting for custom inputs, all features must be supplied in the model's expected order — safest done by editing a DataFrame of averages rather than indexing into `coef_` by hand.
