import sys

import pandas as pd
import sqlalchemy
from mid_analyzer import analyzer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
    predictive_result.sort_values(by="RF Importance", ascending=False, inplace=True)
    rf_imp_html = predictive_result.to_html()
    return rf_imp_html


def main():
    db_user = "root"
    db_pass = "password123"  # pragma: allowlist secret
    db_host = "mariadb:3306"
    db_database = "baseball"
    connect_string = f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"  # pragma
    sql_engine = sqlalchemy.create_engine(connect_string)
    query = """SELECT * FROM model_data"""

    df = pd.read_sql_query(query, sql_engine)
    # df = model_df.toPandas()
    df = df.dropna()
    # df = df.fillna(0)
    # print(df['game_id'])

    df[["game_id", "local_date", "home_team", "away_team"]] = df[
        ["game_id", "local_date", "home_team", "away_team"]
    ].astype(str)
    df[
        [
            "r_home_PA",
            "r_home_BA",
            "r_home_H",
            "r_home_HR",
            "r_home_HR9",
            "r_home_BB",
            "r_home_BB9",
            "r_home_K",
            "r_home_K9",
            "r_home_KBB",
            "r_home_DP",
            "r_home_TP",
            "r_home_SF",
            "r_home_Flyout",
            "r_home_GIDP",
            "r_home_FI",
            "r_home_FE",
            # "r_home_runs",
            # "r_home_errors",
            "r_home_WP",
            "r_away_PA",
            "r_away_BA",
            "r_away_H",
            "r_away_HR",
            "r_away_HR9",
            "r_away_BB",
            "r_away_BB9",
            "r_away_K",
            "r_away_K9",
            "r_away_KBB",
            "r_away_DP",
            "r_away_TP",
            "r_away_SF",
            "r_away_Flyout",
            "r_away_GIDP",
            "r_away_FI",
            "r_away_FE",
            # "r_away_runs",
            # "r_away_errors",
            "r_away_WP",
            "r_diff_PA",
            "r_diff_BA",
            "r_diff_H",
            "r_diff_HR",
            "r_diff_HR9",
            "r_diff_BB",
            "r_diff_BB9",
            "r_diff_K",
            "r_diff_K9",
            "r_diff_KBB",
            "r_diff_DP",
            "r_diff_TP",
            "r_diff_SF",
            "r_diff_Flyout",
            "r_diff_GIDP",
            "r_diff_FI",
            "r_diff_FE",
            # "r_diff_runs",
            # "r_diff_errors",
            "r_diff_WP",
            "temp",
        ]
    ] = df[
        [
            "r_home_PA",
            "r_home_BA",
            "r_home_H",
            "r_home_HR",
            "r_home_HR9",
            "r_home_BB",
            "r_home_BB9",
            "r_home_K",
            "r_home_K9",
            "r_home_KBB",
            "r_home_TP",
            "r_home_DP",
            "r_home_SF",
            "r_home_Flyout",
            "r_home_GIDP",
            "r_home_FI",
            "r_home_FE",
            # "r_home_runs",
            # "r_home_errors",
            "r_home_WP",
            "r_away_PA",
            "r_away_BA",
            "r_away_H",
            "r_away_HR",
            "r_away_HR9",
            "r_away_BB",
            "r_away_BB9",
            "r_away_K",
            "r_away_K9",
            "r_away_KBB",
            "r_away_DP",
            "r_away_TP",
            "r_away_SF",
            "r_away_Flyout",
            "r_away_GIDP",
            "r_away_FI",
            "r_away_FE",
            # "r_away_runs",
            # "r_away_errors",
            "r_away_WP",
            "r_diff_PA",
            "r_diff_BA",
            "r_diff_H",
            "r_diff_HR",
            "r_diff_HR9",
            "r_diff_BB",
            "r_diff_BB9",
            "r_diff_K",
            "r_diff_K9",
            "r_diff_KBB",
            "r_diff_DP",
            "r_diff_TP",
            "r_diff_SF",
            "r_diff_Flyout",
            "r_diff_GIDP",
            "r_diff_FI",
            "r_diff_FE",
            # "r_diff_runs",
            # "r_diff_errors",
            "r_diff_WP",
            "temp",
        ]
    ].astype(
        float
    )

    # Predictors
    pred_cols = [
        "r_home_PA",
        "r_home_BA",
        "r_home_H",
        "r_home_HR",
        "r_home_HR9",
        "r_home_BB",
        "r_home_BB9",
        "r_home_K",
        "r_home_K9",
        "r_home_KBB",
        "r_home_DP",
        "r_home_TP",
        "r_home_SF",
        "r_home_Flyout",
        "r_home_GIDP",
        "r_home_FI",
        "r_home_FE",
        # "r_home_runs",
        # "r_home_errors",
        "r_home_WP",
        "r_away_PA",
        "r_away_BA",
        "r_away_H",
        "r_away_HR",
        "r_away_HR9",
        "r_away_BB",
        "r_away_BB9",
        "r_away_K",
        "r_away_K9",
        "r_away_KBB",
        "r_away_DP",
        "r_away_TP",
        "r_away_SF",
        "r_away_Flyout",
        "r_away_GIDP",
        "r_away_FI",
        "r_away_FE",
        # "r_away_runs",
        # "r_away_errors",
        "r_away_WP",
        "r_diff_PA",
        "r_diff_BA",
        "r_diff_H",
        "r_diff_HR",
        "r_diff_HR9",
        "r_diff_BB",
        "r_diff_BB9",
        "r_diff_K",
        "r_diff_K9",
        "r_diff_KBB",
        "r_diff_DP",
        "r_diff_TP",
        "r_diff_SF",
        "r_diff_Flyout",
        "r_diff_GIDP",
        "r_diff_FI",
        "r_diff_FE",
        # "r_diff_runs",
        # "r_diff_errors",
        "r_diff_WP",
        "temp",
    ]

    # Response (1 as home team wins, 0 as home team loses)
    resp_col = "HomeTeamWins"

    # Reference:
    # https://regenerativetoday.com/four-popular-feature-selection-methods-for-efficient-machine-learning-in-python/
    # Feature Selection
    """
    X = df.drop(columns=["HomeTeamWins", 'game_id', 'local_date', 'home_team', 'away_team'])
    y = df["HomeTeamWins"]
    selected_features = list(X.columns)
    pmax = 1
    while (len(selected_features) > 0):
        p = []
        X_new = X[selected_features]
        X_new = sm.add_constant(X_new)
        model = sm.OLS(y, X_new).fit()
        p = pd.Series(model.pvalues.values[1:], index=selected_features)
        pmax = max(p)
        feature_pmax = p.idxmax()
        if (pmax > 0.05):
            selected_features.remove(feature_pmax)
        else:
            break
    print(selected_features)
    """
    # Reduced Predictor List for Model Building
    # Predictors
    reduced_pred_cols = [
        # "r_home_PA",
        # "r_home_BA",
        # "r_home_H",
        # "r_home_HR",
        # "r_home_HR9",
        # "r_home_BB",
        # "r_home_BB9",
        # "r_home_K",
        # "r_home_K9",
        # "r_home_KBB",
        # "r_home_DP",
        # "r_home_TP",
        # "r_home_SF",
        "r_home_Flyout",
        # "r_home_GIDP",
        # "r_home_FI",
        # "r_home_FE",
        "r_home_WP",
        # "r_away_PA",
        "r_away_BA",
        "r_away_H",
        # "r_away_HR",
        # "r_away_HR9",
        # "r_away_BB",
        # "r_away_BB9",
        # "r_away_K",
        # "r_away_K9",
        "r_away_KBB",
        # "r_away_DP",
        # "r_away_TP",
        # "r_away_SF",
        # "r_away_Flyout",
        # "r_away_GIDP",
        # "r_away_FI",
        # "r_away_FE",
        "r_away_WP",
        # "r_diff_PA",
        # "r_diff_BA",
        # "r_diff_H",
        # "r_diff_HR",
        # "r_diff_HR9",
        # "r_diff_BB",
        # "r_diff_BB9",
        "r_diff_K",
        "r_diff_K9",
        # "r_diff_KBB",
        # "r_diff_DP",
        # "r_diff_TP",
        # "r_diff_SF",
        # "r_diff_Flyout",
        # "r_diff_GIDP",
        # "r_diff_FI",
        "r_diff_FE",
        "r_diff_WP",
        # "temp",
    ]

    # Reference: https://stackoverflow.com/questions/43838052/how-to-get-a-non-shuffled-train-test-split-in-sklearn
    # Historical Data, first 70% as training data, and last 30% as testing data
    train, test = train_test_split(df, test_size=0.3, shuffle=False, random_state=None)

    train_x, train_y = train[reduced_pred_cols], train[resp_col]
    test_x, test_y = test[reduced_pred_cols], test[resp_col]

    model_list = [
        RandomForestClassifier(),
        LogisticRegression(),
        KNeighborsClassifier(5),
        DecisionTreeClassifier(max_depth=4),
        QuadraticDiscriminantAnalysis(),
        SVC(),
        GaussianNB(),
        GradientBoostingClassifier(),
    ]

    model_name = []
    model_score = []
    accu_score = []
    prec_score = []
    f1score = []
    for i in model_list:
        model = Pipeline([("scaler", StandardScaler()), ("classifier", i)])
        model.fit(train_x, train_y)
        model_name.append(str(i))
        model_score.append(str(model.score(test_x, test_y)))
        pred_y = model.predict(test_x)
        accu_score.append(str(accuracy_score(test_y, pred_y)))
        prec_score.append(str(precision_score(test_y, pred_y)))
        f1score.append(str(f1_score(test_y, pred_y)))
    data = {
        "Model/Method": model_name,
        "Score": model_score,
        "Accuracy Score": accu_score,
        "Precision Score": prec_score,
        "F1 Score": f1score,
    }
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

    # Print the best scores
    print(data)


if __name__ == "__main__":
    sys.exit(main())
