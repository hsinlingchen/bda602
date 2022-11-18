import sys

import pandas as pd
from mid_analyzer import analyzer
from pyspark.sql import SparkSession
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# Random Forest Feature Importance Rank
def rf_imp_rank(df, pred_cols, resp):
    df_X = df[pred_cols]
    df_y = df[resp]
    rf_c = RandomForestRegressor(max_depth=2, random_state=0)
    rf_c.fit(df_X.values, df_y)
    imp = rf_c.feature_importances_
    imp_list = imp.tolist()
    data = {"Predictor": pred_cols, "RF Importance": imp_list}
    predictive_result = pd.DataFrame(data)
    rf_imp_html = predictive_result.to_html()
    return rf_imp_html


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    user = "root"
    password = "root"  # pragma: allowlist secret
    server = "localhost"
    database = "baseball"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"
    sql_query = """SELECT * FROM model_data"""
    model_df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql_query)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )
    df = model_df.toPandas()
    df = df.dropna()
    df[["game_id", "home_team", "away_team", "local_date"]] = df[
        ["game_id", "home_team", "away_team", "local_date"]
    ].astype(str)
    df[["home_BA", "away_BA"]] = df[["home_BA", "away_BA"]].astype(float)
    df[["home_PA", "home_H", "away_PA", "away_H"]] = df[
        ["home_PA", "home_H", "away_PA", "away_H"]
    ].astype(int)
    # print(df.dtypes)

    # Predictors
    pred_cols = [
        "home_PA",
        "home_BA",
        "home_H",
        "home_HR",
        "home_BB",
        "home_K",
        "home_TP",
        "home_Flyout",
        "home_GIDP",
        "away_PA",
        "away_BA",
        "away_H",
        "away_HR",
        "away_BB",
        "away_K",
        "away_TP",
        "away_Flyout",
        "away_GIDP",
    ]
    # Response (1 as home team wins, 0 as home team loses)
    resp_col = "HomeTeamWins"

    # Reference: https://stackoverflow.com/questions/43838052/how-to-get-a-non-shuffled-train-test-split-in-sklearn
    # Historical Data, first 80% as training data, and last 20% as testing data
    train, test = train_test_split(df, test_size=0.20, shuffle=False, random_state=None)

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

    rf_imp = rf_imp_rank(df, pred_cols, resp_col)
    # Applying hw4 and midterm analyzer
    analyzer(df, pred_cols, resp_col)
    file = open("report.html", "a")
    file.write(pr_html)
    file.write(rf_imp)
    file.close()

    # Model Fitting Result:
    # R^2 scores show that QDA is the highest among five models,
    # that it provides the best predictive result of all; even though 54.67% still has lots of room for improvement.


if __name__ == "__main__":
    sys.exit(main())
