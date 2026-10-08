# Polynomial Regression

This repository contains the implementation for polynomial regression on two
personalised datasets. Polynomial models are evaluated using 5-fold
cross-validation, and Linear Regression, Ridge Regression, and Lasso Regression
are compared to select the best-performing approach.

## Dataset

Two datasets are used:

- `var1`: 6 input features, polynomial degrees 1 to 10
- `var2`: 3 input features, polynomial degrees 1 to 20

## Method

For each polynomial degree, the following approaches are evaluated:

1. Polynomial + Linear Regression
2. Polynomial + Ridge Regression
3. Polynomial + Lasso Regression

For Ridge and Lasso, the following regularisation strengths are tested:

```text
alpha = 0.01, 0.1, 1, 10
