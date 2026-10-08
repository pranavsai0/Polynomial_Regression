import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate

train_data = pd.read_csv("BT2024099_train_var2.csv")
test_data = pd.read_csv("BT2024099_test_var2.csv")

X_train = train_data.drop("y", axis=1)
y_train = train_data["y"]
X_test = test_data[X_train.columns]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

alphas = [0.01, 0.1, 1, 10]

best_mse = float("inf")
best_model_name = None
best_degree = None
best_alpha = None
best_r2 = None

linear_mse = []
ridge_mse = []
lasso_mse = []

print("Testing models for var2...\n")

# Polynomial + Linear Regression
for degree in range(1, 21):

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
        n_jobs=-1
    )

    mse = -scores["test_mse"].mean()
    r2 = scores["test_r2"].mean()

    linear_mse.append(mse)

    print(f"Linear | Degree {degree}: "
          f"MSE = {mse:.6f}, R2 = {r2:.6f}")

    if mse < best_mse:
        best_mse = mse
        best_model_name = "Linear"
        best_degree = degree
        best_alpha = None
        best_r2 = r2


# Polynomial + Ridge
for degree in range(1, 21):

    degree_best_mse = float("inf")

    for alpha in alphas:

        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            Ridge(alpha=alpha)
        )

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
            n_jobs=-1
        )

        mse = -scores["test_mse"].mean()
        r2 = scores["test_r2"].mean()

        print(f"Ridge  | Degree {degree}, Alpha {alpha}: "
              f"MSE = {mse:.6f}, R2 = {r2:.6f}")

        if mse < degree_best_mse:
            degree_best_mse = mse

        if mse < best_mse:
            best_mse = mse
            best_model_name = "Ridge"
            best_degree = degree
            best_alpha = alpha
            best_r2 = r2

    ridge_mse.append(degree_best_mse)


# Polynomial + Lasso
for degree in range(1, 21):

    degree_best_mse = float("inf")

    for alpha in alphas:

        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            Lasso(alpha=alpha, max_iter=20000, tol=1e-3)
        )

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
            n_jobs=-1
        )

        mse = -scores["test_mse"].mean()
        r2 = scores["test_r2"].mean()

        print(f"Lasso  | Degree {degree}, Alpha {alpha}: "
              f"MSE = {mse:.6f}, R2 = {r2:.6f}")

        if mse < degree_best_mse:
            degree_best_mse = mse

        if mse < best_mse:
            best_mse = mse
            best_model_name = "Lasso"
            best_degree = degree
            best_alpha = alpha
            best_r2 = r2

    lasso_mse.append(degree_best_mse)


print("\nBest model for var2")
print("Model:", best_model_name)
print("Degree:", best_degree)
print("Alpha:", best_alpha)
print(f"CV MSE: {best_mse:.6f}")
print(f"CV R2 : {best_r2:.6f}")


# Comparison graph
degrees = list(range(1, 21))

plt.figure(figsize=(10, 5))

plt.plot(degrees, linear_mse, marker="o", label="Linear")
plt.plot(degrees, ridge_mse, marker="o", label="Ridge")
plt.plot(degrees, lasso_mse, marker="o", label="Lasso")

plt.scatter(
    best_degree,
    best_mse,
    s=120,
    label=f"Best: {best_model_name}, Degree {best_degree}"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Cross-Validation MSE (log scale)")
plt.title("var2: Comparison of Linear, Ridge and Lasso")
plt.xticks(degrees)
plt.yscale("log")
plt.grid(True, which="both")
plt.legend()

plt.tight_layout()
plt.savefig("var2_method_comparison.png", dpi=300)
plt.show()


# Train final model
if best_model_name == "Linear":

    final_model = make_pipeline(
        PolynomialFeatures(degree=best_degree, include_bias=False),
        LinearRegression()
    )

elif best_model_name == "Ridge":

    final_model = make_pipeline(
        PolynomialFeatures(degree=best_degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=best_alpha)
    )

else:

    final_model = make_pipeline(
        PolynomialFeatures(degree=best_degree, include_bias=False),
        StandardScaler(),
        Lasso(alpha=best_alpha, max_iter=20000, tol=1e-3)
    )


final_model.fit(X_train, y_train)

predictions = final_model.predict(X_test)

pd.DataFrame({"y": predictions}).to_csv(
    "BT2024099_pred_var2.csv",
    index=False
)

print("\nPrediction file saved:")
print("BT2024099_pred_var2.csv")
