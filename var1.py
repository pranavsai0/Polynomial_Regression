import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate


# Load the training and test datasets
train_data = pd.read_csv("BT2024099_train_var1.csv")
test_data = pd.read_csv("BT2024099_test_var1.csv")

X_train = train_data.drop("y", axis=1)
y_train = train_data["y"]
X_test = test_data[X_train.columns]

# Use 5-fold cross-validation to choose the polynomial degree
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_degree = 1
best_mse = float("inf")
best_r2 = None

degrees = []
mse_values = []
r2_values = []

print("Testing polynomial degrees...\n")

for degree in range(1, 11):

    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring={
            "mse": "neg_mean_squared_error",
            "r2": "r2"
        }
    )

    mse = -scores["test_mse"].mean()
    r2 = scores["test_r2"].mean()

    degrees.append(degree)
    mse_values.append(mse)
    r2_values.append(r2)

    print(
        f"Degree {degree}: "
        f"CV MSE = {mse:.6f}, "
        f"CV R2 = {r2:.6f}"
    )

    # Select the degree with the lowest validation MSE
    if mse < best_mse:
        best_mse = mse
        best_degree = degree
        best_r2 = r2

print("\nSelected model for var1")
print("Best degree:", best_degree)
print(f"Best CV MSE: {best_mse:.6f}")
print(f"Best CV R2 : {best_r2:.6f}")

# Plot CV MSE and CV R2 against polynomial degree
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(
    degrees,
    mse_values,
    marker="o",
    label="CV MSE"
)
ax1.set_xlabel("Polynomial Degree")
ax1.set_ylabel("CV MSE")
ax1.set_xticks(degrees)
ax1.grid(True)

ax2 = ax1.twinx()

ax2.plot(
    degrees,
    r2_values,
    marker="o",
    linestyle="--",
    label="CV R2"
)
ax2.set_ylabel("CV R2")

ax1.scatter(
    best_degree,
    best_mse,
    s=100,
    label=f"Best degree = {best_degree}"
)

ax2.scatter(
    best_degree,
    best_r2,
    s=100
)

plt.title("var1: CV MSE and CV R2 vs Polynomial Degree")
fig.tight_layout()
plt.show()

# Refit the selected model using all training data
final_model = make_pipeline(
    PolynomialFeatures(
        degree=best_degree,
        include_bias=False
    ),
    LinearRegression()
)

final_model.fit(X_train, y_train)

# Generate predictions for the test dataset
predictions = final_model.predict(X_test)

# Submission file should contain only the predicted y values
pd.DataFrame({"y": predictions}).to_csv(
    "BT2024099_pred_var1.csv",
    index=False
)

print("\nPredictions saved to:")
print("BT2024099_pred_var1.csv")