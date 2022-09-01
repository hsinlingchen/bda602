import sys
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    #Load iris data into DataFrame using pandas
    iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    iris = pd.read_csv("https://teaching.mrsharky.com/data/iris.data", names=iris_names)


    #simple summary statistics
    stat = iris.iloc[:, 0:4]
    stat_mean = stat.mean()
    print("Mean:\n{}".format(stat_mean))
    stat_min = stat.min()
    print("Min:\n{}".format(stat_min))
    stat_max = stat.max()
    print("Max:\n{}".format(stat_max))
    q_25 = np.quantile(stat, 0.25)
    q_50 = np.quantile(stat, 0.5)
    q_75 = np.quantile(stat, 0.75)
    print("Quantiles:\n0.25 = {}\n0.50 = {}\n0.75 = {}".format(q_25, q_50, q_75))


    #plots
    #scatter plots for sepal length and width
    sepal_scatter = px.scatter(iris, x="sepal length", y="sepal width", color="class")
    sepal_scatter.write_html(file="sepal_scatter.html", include_plotlyjs="cdn", auto_open=True)
    #scatter plots for petal length and width
    petal_scatter = px.scatter(iris, x="petal length", y="petal width", color="class")
    petal_scatter.write_html(file="petal_scatter.html", include_plotlyjs="cdn", auto_open=True)
    #violin plot for sepal length
    sl_violin = px.violin(iris, y="sepal length", color="class")
    sl_violin.write_html(file="sepal_length_violin.html", include_plotlyjs="cdn", auto_open=True)
    #violin plot for petal length
    sw_violin = px.violin(iris, y="sepal width", color="class")
    sw_violin.write_html(file="sepal_width_violin.html", include_plotlyjs="cdn", auto_open=True)
    #histogram for sepal length
    hist_fig = px.histogram(iris, x="sepal length", y="sepal width", color="class")
    hist_fig.write_html(file="sepal_hist.html", include_plotlyjs="cdn", auto_open=True)


    #Analyze and build models
    # Increase pandas print viewport (so we see more on the screen)
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    # DataFrame to numpy values
    X_orig = iris.iloc[:, 0:4]
    y = iris["class"].values

    # Let's generate a feature from the where they started
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(X_orig)
    X = one_hot_encoder.transform(X_orig)

    # Fit the features to a random forest
    random_forest = RandomForestClassifier(random_state=1234)
    random_forest.fit(X, y)


    prediction = random_forest.predict(X)
    probability = random_forest.predict_proba(X)

    print("Model Predictions")
    print(f"Classes: {random_forest.classes_}")
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")

    # As pipeline
    print("Model via Pipeline Predictions")
    pipeline = Pipeline(
        [
            ("OneHotEncode", OneHotEncoder()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline.fit(X_orig, y)

    probability = pipeline.predict_proba(X_orig)
    prediction = pipeline.predict(X_orig)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")
    return


if __name__ == "__main__":
    sys.exit(main())