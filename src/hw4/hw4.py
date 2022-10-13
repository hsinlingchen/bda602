import sys

import pandas as pd
import statsmodels.api
from plotly import express as px
from sklearn.ensemble import RandomForestClassifier

# Difference with mean of response along with it's plot (weighted and unweighted)
# Continuous Predictor
"""
d


# Random Forest Feature Importance
def rf_cls(predictor, response):
    df_X = predictor[column]
    df_y = response
    rf_c = RandomForestClassifier(max_depth=2, random_state=0)
    rf_c.fit(df_X.values.reshape(-1, 1), df_y)
    imp = rf_c.feature_importances_
    return print(imp)

def rf_reg(predictor, response):
    df_X = predictor[column]
    df_y = response
    rf_c = RandomForestRegressor(max_depth=2, random_state=0)
    rf_c.fit(df_X.values.reshape(-1, 1), df_y)
    imp = rf_c.feature_importances_
    return print(imp)
"""


def main():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    ).dropna()
    # Predictors
    pred_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
    pred = df[pred_cols]
    # Response (1 as survived, 0 as not survived)
    resp_col = "survived"
    resp = df[resp_col]
    # create a list for continuous predictor for further random forest importance
    cont_pred_list = list()
    # determine if the response is continuous, boolean, or categorical
    if float(resp.nunique()) / resp.count() < 0.05:
        print("Response is boolean.")
    elif resp is str:
        print("Response is categorical.")
    else:
        print("Response is continuous.")

    # determine if the predictors are continuous, boolean, or categorical
    for column in pred:
        if pred[column].dtype.name in ["object", "int64"]:
            print(f"Predictor {column} is categorical.")
            # plots for boolean response with categorical predictor
            pn = pred[column].name
            rn = resp.name
            df = pd.DataFrame({pn: pred[column], rn: resp})
            fig_h = px.density_heatmap(
                df,
                x=pn,
                y=rn,
                title=f"Predictor {pred[column].name} vs. Response {resp.name}",
            )
            fig_h.show()
        elif float(pred[column].nunique()) / pred[column].count() < 0.05:
            print(f"Predictor {column} is boolean.")
            # plots for boolean response with boolean predictor
            pn = pred[column].name
            rn = resp.name
            df = pd.DataFrame({pn: pred[column], rn: resp})
            fig_h = px.density_heatmap(
                df,
                x=pn,
                y=rn,
                title=f"Predictor {pred[column].name} vs. Response {resp.name}",
            )
            fig_h.show()
        else:
            print(f"Predictor {column} is continuous.")
            cont_pred_list.append(pred[column].name)
            # plots for boolean response with continuous predictor
            df_cont = pred[column].values
            fig = px.histogram(
                df_cont,
                color=resp,
                marginal="rug",
                title=f"Predictor {pred[column].name} vs. Response {resp.name}",
            )
            fig.show()
            fig1 = px.violin(
                df_cont,
                color=resp,
            )
            fig1.show()

            # logistic_regression
            feature_name = pred[column].name
            p = statsmodels.api.add_constant(pred[column].values)
            log_reg = statsmodels.api.Logit(resp.values, p)
            log_reg_fit = log_reg.fit()
            t_val = round(log_reg_fit.tvalues[1], 6)
            p_val = "{:.6e}".format(log_reg_fit.pvalues[1])
            print(f"t-value: {t_val}\np-value: {p_val}")
            # Plot the figure
            fig_lr = px.scatter(x=pred[column], y=resp, trendline="lowess")
            fig_lr.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_val}) (p-value={p_val})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title="y",
            )
            fig_lr.show()
            # Difference Mean of response (Unweighted) ...

            # Difference Mean of response (Weighted) ...

    # Random Forest Feature Importance Rank
    df_X = pred[cont_pred_list]
    df_y = resp
    rf_c = RandomForestClassifier(max_depth=2, random_state=0)
    rf_c.fit(df_X.values, df_y)
    imp = rf_c.feature_importances_
    print(imp)

    # table report....


if __name__ == "__main__":
    sys.exit(main())
