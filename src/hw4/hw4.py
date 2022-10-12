# import sys


import pandas as pd
import statsmodels.api
from plotly import express as px
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
).dropna()
# Predictors
pred_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
pred = df[pred_cols]
# Response (1 as survived, 0 as not survived)
resp = df["survived"]


# determine if the response is continuous, boolean, or categorical
if float(resp.nunique()) / resp.count() < 0.05:
    print("Response is boolean.")
elif resp is str:
    print("Response is categorical.")
else:
    print("Response is continuous.")


# plots for boolean response
def plot_cat_bool(predictor, response):
    pn = predictor[column].name
    rn = response.name
    df = pd.DataFrame({pn: pred[column], rn: resp})
    fig_h = px.density_heatmap(
        df, x=pn, y=rn, title=f"Predictor {pred[column].name} vs. Response {resp.name}"
    )
    fig_h.show()


def plot_con_bool(predictor, response):
    df = predictor[column].values
    fig = px.histogram(
        df,
        color=response,
        marginal="rug",
        title=f"Predictor {predictor[column].name} vs. Response {response.name}",
    )
    fig.show()
    fig1 = px.violin(
        df,
        color=response,
        title=f"Predictor {predictor[column].name} vs. Response {response.name}",
    )
    fig1.show()


# Logistic Regression: Boolean response
def logistic_regression(predictor, response):
    feature_name = predictor[column].name
    p = statsmodels.api.add_constant(predictor[column].values)
    log_reg = statsmodels.api.Logit(response.values, p)
    log_reg_fit = log_reg.fit()
    t_val = round(log_reg_fit.tvalues[1], 6)
    p_val = "{:.6e}".format(log_reg_fit.pvalues[1])
    print(f"t-value: {t_val}\np-value: {p_val}")
    # Plot the figure
    fig = px.scatter(x=predictor[column], y=resp, trendline="lowess")
    fig.update_layout(
        title=f"Variable: {feature_name}: (t-value={t_val}) (p-value={p_val})",
        xaxis_title=f"Variable: {feature_name}",
        yaxis_title="y",
    )
    fig.show()
    return


# Difference with mean of response along with it's plot (weighted and unweighted)


# Random Forest Feature Importance
def rf_reg(predictor, response):
    df_X = predictor[column]
    df_y = response
    rf_c = RandomForestClassifier(max_depth=2, random_state=0)
    rf_c.fit(df_X.values.reshape(-1, 1), df_y)
    imp = rf_c.feature_importances_
    return imp


# determine if the predictors are continuous, boolean, or categorical
for column in pred:
    if pred[column].dtype.name in ["object", "int64"]:
        print(f"Predictor {column} is categorical.")
        # plot_cat_bool(pred, resp)
    elif float(pred[column].nunique()) / pred[column].count() < 0.05:
        print(f"Predictor {column} is boolean.")
        # plot_cat_bool(pred, resp)
    else:
        print(f"Predictor {column} is continuous.")
        # plot_con_bool(pred, resp)
        # logistic_regression(pred, resp)
        rf_reg(pred, resp)
