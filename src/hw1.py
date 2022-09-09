import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main():
    # Load iris data into DataFrame using pandas
    iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    iris = pd.read_csv("./src/iris.data", names=iris_names)

    # simple summary statistics
    stat = iris.iloc[:, 0:4]
    stat_mean = stat.mean()
    print("Mean:\n{}".format(stat_mean))
    stat_min = stat.min()
    print("Min:\n{}".format(stat_min))
    stat_max = stat.max()
    print("Max:\n{}".format(stat_max))
    q_sl_25 = np.quantile(stat["sepal length"], 0.25)
    q_sl_50 = np.quantile(stat["sepal length"], 0.5)
    q_sl_75 = np.quantile(stat["sepal length"], 0.75)
    print(
        "Quartiles of Sepal Length:\n0.25 = {}\n0.50 = {}\n0.75 = {}".format(
            q_sl_25, q_sl_50, q_sl_75
        )
    )
    q_sw_25 = np.quantile(stat["sepal width"], 0.25)
    q_sw_50 = np.quantile(stat["sepal width"], 0.5)
    q_sw_75 = np.quantile(stat["sepal width"], 0.75)
    print(
        "Quartiles of Sepal Width:\n0.25 = {}\n0.50 = {}\n0.75 = {}".format(
            q_sw_25, q_sw_50, q_sw_75
        )
    )
    q_pl_25 = np.quantile(stat["petal length"], 0.25)
    q_pl_50 = np.quantile(stat["petal length"], 0.5)
    q_pl_75 = np.quantile(stat["petal length"], 0.75)
    print(
        "Quartiles of Petal Length:\n0.25 = {}\n0.50 = {}\n0.75 = {}".format(
            q_pl_25, q_pl_50, q_pl_75
        )
    )
    q_pw_25 = np.quantile(stat["petal width"], 0.25)
    q_pw_50 = np.quantile(stat["petal width"], 0.5)
    q_pw_75 = np.quantile(stat["petal width"], 0.75)
    print(
        "Quartiles of Petal Width:\n0.25 = {}\n0.50 = {}\n0.75 = {}".format(
            q_pw_25, q_pw_50, q_pw_75
        )
    )

    # plots
    # scatter plots for sepal length and width
    sepal_scatter = px.scatter(iris, x="sepal length", y="sepal width", color="class")
    sepal_scatter.write_html(
        file="sepal_scatter.html", include_plotlyjs="cdn", auto_open=True
    )
    # scatter plots for petal length and width
    petal_scatter = px.scatter(iris, x="petal length", y="petal width", color="class")
    petal_scatter.write_html(
        file="petal_scatter.html", include_plotlyjs="cdn", auto_open=True
    )
    # violin plot for sepal length
    sl_violin = px.violin(iris, y="sepal length", color="class")
    sl_violin.write_html(
        file="sepal_length_violin.html", include_plotlyjs="cdn", auto_open=True
    )
    # violin plot for petal length
    sw_violin = px.violin(iris, y="sepal width", color="class")
    sw_violin.write_html(
        file="sepal_width_violin.html", include_plotlyjs="cdn", auto_open=True
    )
    # histogram for sepal length
    hist_fig = px.histogram(iris, x="sepal length", y="sepal width", color="class")
    hist_fig.write_html(file="sepal_hist.html", include_plotlyjs="cdn", auto_open=True)

    # Analyze and build models
    # DataFrame to numpy values
    X_orig = iris.iloc[:, 0:4].values
    y = iris["class"].values

    # As pipeline
    # Random Forest
    print("Model via Random Forest Predictions")
    pipeline_rf = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline_rf.fit(X_orig, y)

    prob_rf = pipeline_rf.predict_proba(X_orig)
    pred_rf = pipeline_rf.predict(X_orig)
    print(f"Probability: {prob_rf}")
    print(f"Predictions: {pred_rf}")
    # Decision Tree
    print("Model via Decision Tree Predictions")
    pipeline_dt = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("DecisionTree", DecisionTreeClassifier(random_state=1234)),
        ]
    )
    pipeline_dt.fit(X_orig, y)

    prob_dt = pipeline_dt.predict_proba(X_orig)
    pred_dt = pipeline_dt.predict(X_orig)
    print(f"Probability: {prob_dt}")
    print(f"Predictions: {pred_dt}")
    # K Neighbors
    print("Model via K Neighbors Predictions")
    pipeline_kn = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("KNeighbors", KNeighborsClassifier(n_neighbors=5)),
        ]
    )
    pipeline_kn.fit(X_orig, y)

    prob_kn = pipeline_kn.predict_proba(X_orig)
    pred_kn = pipeline_kn.predict(X_orig)
    print(f"Probability: {prob_kn}")
    print(f"Predictions: {pred_kn}")
    return


if __name__ == "__main__":
    sys.exit(main())
