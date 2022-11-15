import sys

import pandas as pd
from mid_analyzer import analyzer
from pyspark.sql import SparkSession
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    user = "root"
    password = "root"  # pragma: allowlist secret
    server = "localhost"
    database = "baseball"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"
    sql_query = """SELECT * FROM pitching_data"""
    pitching_df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql_query)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )
    df = pitching_df.toPandas()
    df = df.dropna()
    df = df.astype({"H": "float64", "PIT": "float64"})
    # print(df.dtypes)

    # Predictors
    pred_cols = ["IP", "H", "GIDP", "HR", "K", "BB", "PIT", "TP", "WHIP", "PFR"]
    # Response (1 as home team wins, 0 as home team loses)
    resp_col = "HomeTeamWins"

    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    train, test = train_test_split(df, test_size=0.30, random_state=42)
    # train = df.iloc[:-1]
    # test = df.iloc[-1:]

    train_x, train_y = train[pred_cols], train[resp_col]
    test_x, test_y = test[pred_cols], test[resp_col]

    model_list = [
        RandomForestClassifier(),
        KNeighborsClassifier(5),
        DecisionTreeClassifier(max_depth=4),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression(),
    ]

    model_name = []
    model_score = []
    for i in model_list:
        model = Pipeline([("scaler", StandardScaler()), ("classifier", i)])
        model.fit(train_x, train_y)
        model_name.append(str(i))
        model_score.append(str(model.score(test_x, test_y)))
    data = {"Model/Method": model_name, "Score": model_score}
    predictive_result = pd.DataFrame(data)
    pr_html = predictive_result.to_html()
    # https://stackoverflow.com/questions/24458163/what-are-the-parameters-for-sklearns-score-function

    # Applying midterm project to the data
    analyzer(df, pred_cols, resp_col)
    file = open("report.html", "a")
    file.write(pr_html)
    file.close()

    # Model Fitting Result:
    # R^2 scores show that LogisticRegression is the highest among five models,
    # that it provides the best predictive result of all; even though 53.3% still has lots of room for improvement.


if __name__ == "__main__":
    sys.exit(main())
