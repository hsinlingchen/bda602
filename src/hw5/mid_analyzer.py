import itertools
import os
import warnings
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api
from diff_w_mean import cat_cat_diff, cat_diff, con_cat_diff, con_con_diff, con_diff
from plot import con_cat_plot, corr_plot, heatmap
from plotly import express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor


def response_type(df, resp_col):
    resp = df[resp_col]
    if len(resp.unique()) == 2:
        r_type = "boolean"
        print(f"{resp_col} is {r_type}")
    else:
        r_type = "continuous"
        print(f"{resp_col} is {r_type}")


# split predictor (cat/con)
cat_pred = []
con_pred = []


def predictor_type(df, pred_cols):
    for column in df[pred_cols]:
        x = df[pred_cols][column]
        # print(x.dtype.name)
        if x.dtype == float or x.dtype == int:
            con_pred.append(column)
        else:
            cat_pred.append(column)
    return cat_pred, con_pred


def make_clickable(value):
    if value != "NA":
        return f'<a target="_blank" href="{value}">{value}</a>'
    else:
        return "NA"


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cat_cont_correlation_ratio(categories, values):
    """
    source: week 7 lecture
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def cat_cat_corr(x, y, bias_correction=True, tschuprow=False):
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(
                0,
                (phi2 - ((r - 1) * (c - 1)) / (n_observations - 1)),
            )
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected
                    / np.sqrt(
                        ((r_corrected - 1) * (c_corrected - 1)),
                    )
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def linear_regression(df, predictor, response):
    """
    Source: lecture 6
    """
    predictor_for_model = statsmodels.api.add_constant(df[predictor])
    linear_regression_model = statsmodels.api.OLS(df[response], predictor_for_model)
    linear_regression_model_fitted = linear_regression_model.fit()

    t_value = linear_regression_model_fitted.tvalues[1]
    p_value = linear_regression_model_fitted.pvalues[1]

    linear_reg_plot = "output/" + f"{predictor}_{response}_linear_regression.html"

    fig = px.scatter(
        data_frame=df,
        x=predictor,
        y=response,
        trendline="ols",
        title=f"(t-value={t_value}) (p-value={p_value})",
    )

    fig.write_html(
        file=linear_reg_plot,
        include_plotlyjs="cdn",
    )

    return linear_reg_plot


def rf_imp(df, predictors, response):
    df_pred = df[predictors].astype("float64")
    # reference: https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown
    df_resp = df[response]
    rf_reg = RandomForestRegressor(max_depth=2, random_state=0)
    # reference:
    # https://stackoverflow.com/questions/47942417/valueerror-unknown-label-type-continuous-when-applying-random-forrest
    rf_reg.fit(df_pred, df_resp)
    imp = rf_reg.feature_importances_
    return imp.tolist()


def analyzer(df, pred_cols, resp_col):
    # create variables for html outputs
    # corr_df_html = ""
    con_con_df_html = ""
    con_con_diff_styler = ""
    full_pred_df_html = ""
    con_cat_corr_df_html = ""
    con_cat_diff_html = ""
    cat_cat_html_styler = ""
    cat_cat_diff_styler = ""

    # make an directory for plot outputs
    out_dir_exist = os.path.exists("output")
    if out_dir_exist is False:
        os.makedirs("output")

    # load predictors data sorting types
    predictor_type(df, pred_cols)

    # Continuous / Continuous (Linear Regression)
    con_comb_list = list(itertools.combinations(con_pred, r=2))
    con_con_df = pd.DataFrame(
        columns=[
            "Predictors",
            "Pearson's r",
            "Absolute Value of Correlation",
            "Linear Regression Plot",
        ]
    )
    for pred1, pred2 in con_comb_list:
        lr_plot = linear_regression(df, pred1, pred2)
        con_pearson = stats.pearsonr(df[pred1], df[pred2])[0]
        # con_p = stats.pearsonr(df[pred1], df[pred2])[1]
        new_row = {
            "Predictors": f"{pred1} and {pred2}",
            "Pearson's r": con_pearson,
            "Absolute Value of Correlation": abs(con_pearson),
            "Linear Regression Plot": lr_plot,
        }
        con_con_df = con_con_df.append(new_row, ignore_index=True)
        con_con_df = con_con_df.sort_values(
            by="Absolute Value of Correlation", ascending=False
        )
        con_con_df_html = con_con_df.style.format(
            {"Linear Regression Plot": make_clickable}
        ).to_html()

    con_con_diff_df = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Residual Plot",
        ]
    )
    for pred1, pred2 in con_comb_list:
        file1, file2, msd, wmsd = con_con_diff(df, pred1, pred2, resp_col)
        new_row = {
            "Predictor 1": pred1,
            "Predictor 2": pred2,
            "Difference of Mean Response": msd,
            "Weighted Difference of Mean Response": wmsd,
            "Bin Plot": file1,
            "Residual Plot": file2,
        }
        con_con_diff_df = con_con_diff_df.append(new_row, ignore_index=True)
        con_con_diff_df = con_con_diff_df.sort_values(
            by="Difference of Mean Response", ascending=False
        )
        con_con_diff_styler = con_con_diff_df.style.format(
            {
                "Bin Plot": make_clickable,
                "Residual Plot": make_clickable,
            }
        )
        con_con_diff_styler = con_con_diff_styler.to_html()

    # Continuous / Categorical
    rf = rf_imp(df, con_pred, resp_col)
    con_cat_corr_d = pd.DataFrame(columns=cat_pred, index=con_pred)
    con_cat_corr_df = pd.DataFrame(
        columns=[
            "Predictors",
            "Correlation Ratio",
            "Absolute Value of Correlation",
            "Distribution Plot",
            "Violin Plot",
        ]
    )
    con_cat_diff_df = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Residual Plot",
        ]
    )

    full_pred_diff_df = pd.DataFrame(
        columns=[
            "Predictor",
            # "Random Forest Importance",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Response Plot",
        ]
    )

    i = 0
    for pred1 in con_pred:
        # p_value, t_value = logistic_regression(df, pred1, resp_col)
        con_diff_plot, msd, wmsd = con_diff(df, pred1, resp_col)
        con_cat_hist, con_cat_vio = con_cat_plot(df, pred1, resp_col)
        new_row = {
            "Predictor": pred1,
            # "p-value": p_value,
            # "t-value": t_value,
            "Random Forest Importance": rf[i],
            "Difference of Mean Response": msd,
            "Weighted Difference of Mean Response": wmsd,
            "Bin Plot": con_diff_plot,
            "Response Plot": con_cat_hist,
        }
        full_pred_diff_df = full_pred_diff_df.append(new_row, ignore_index=True)
        i += 1
        for pred2 in cat_pred:
            cat_diff_plot, msd, wmsd = cat_diff(df, pred2, resp_col)
            cat_diff_hm = heatmap(df, pred2, resp_col)
            new_row = {
                "Predictor": pred2,
                "Difference of Mean Response": msd,
                "Weighted Difference of Mean Response": wmsd,
                "Bin Plot": cat_diff_plot,
                "Response Plot": cat_diff_hm,
            }
            full_pred_df = full_pred_diff_df.append(new_row, ignore_index=True)
            full_pred_df = full_pred_df.sort_values(
                by=["Weighted Difference of Mean Response"], ascending=False
            )
            full_pred_df_html = full_pred_df.style.format(
                {
                    "Bin Plot": make_clickable,
                    "Response Plot": make_clickable,
                }
            ).to_html()

            con_cat_dist, con_cat_vio = con_cat_plot(df, pred1, pred2)
            pred1_array = df[pred1].to_numpy()
            pred2_array = df[pred2].to_numpy()
            con_cat_corr = cat_cont_correlation_ratio(pred2_array, pred1_array)
            con_cat_corr_d.at[pred1, pred2] = con_cat_corr
            new_row = {
                "Predictors": f"{pred1} and {pred2}",
                "Correlation Ratio": con_cat_corr,
                "Absolute Value of Correlation": abs(con_cat_corr),
                "Violin Plot": con_cat_vio,
                "Distribution Plot": con_cat_dist,
            }
            con_cat_corr_tb = con_cat_corr_df.append(new_row, ignore_index=True)
            con_cat_corr_df = con_cat_corr_tb.sort_values(
                by=["Absolute Value of Correlation"], ascending=False
            )
            con_cat_corr_df_html = con_cat_corr_df.style.format(
                {
                    "Violin Plot": make_clickable,
                    "Distribution Plot": make_clickable,
                }
            ).to_html()

            diff_bin, residual_plot, diff, w_diff = con_cat_diff(
                df, pred1, pred2, resp_col
            )
            new_row = {
                "Predictor 1": pred1,
                "Predictor 2": pred2,
                "Difference of Mean Response": diff,
                "Weighted Difference of Mean Response": w_diff,
                "Bin Plot": diff_bin,
                "Residual Plot": residual_plot,
            }
            con_cat_diff_df = con_cat_diff_df.append(new_row, ignore_index=True)
            con_cat_diff_sort = con_cat_diff_df.sort_values(
                by=["Difference of Mean Response"], ascending=False
            )
            con_cat_diff_html = con_cat_diff_sort.style.format(
                {
                    "Bin Plot": make_clickable,
                    "Residual Plot": make_clickable,
                }
            ).to_html()

    # Categorical / Categorical
    comb_list = list(itertools.combinations(cat_pred, r=2))
    cat_cat_corr_df = pd.DataFrame(columns=cat_pred, index=cat_pred)
    # print(comb_list)
    cat_cat_df = pd.DataFrame(
        columns=[
            "Predictors",
            "Cramer's V",
            "Absolute Value of Correlation",
            "Heatmap",
        ]
    )
    for (
        pred1,
        pred2,
    ) in comb_list:
        hm_link = heatmap(df, pred1, pred2)
        corr = cat_cat_corr(df[pred1], df[pred2])
        new_row = {
            "Predictors": f"{pred1} and {pred2}",
            "Cramer's V": corr,
            "Absolute Value of Correlation": abs(corr),
            "Heatmap": hm_link,
        }
        cat_cat_df = cat_cat_df.append(new_row, ignore_index=True)
        cat_cat_df = cat_cat_df.sort_values(
            by=["Absolute Value of Correlation"], ascending=False
        )
        hm_styler = cat_cat_df.style.format({"Heatmap": make_clickable})
        cat_cat_html_styler = hm_styler.to_html()
        cat_cat_c = cat_cat_corr(df[pred1], df[pred2])
        cat_cat_corr_df.at[pred1, pred2] = cat_cat_c
        cat_cat_corr_df.at[pred2, pred1] = cat_cat_c

    cat_cat_diff_df = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Residual Plot",
        ]
    )
    for (
        pred1,
        pred2,
    ) in comb_list:
        # Bruce Force
        file_hm, file_res, msd, wmsd = cat_cat_diff(df, pred1, pred2, resp_col)
        new_row = {
            "Predictor 1": pred1,
            "Predictor 2": pred2,
            "Difference of Mean Response": msd,
            "Weighted Difference of Mean Response": wmsd,
            "Bin Plot": file_hm,
            "Residual Plot": file_res,
        }
        cat_cat_diff_df = cat_cat_diff_df.append(new_row, ignore_index=True)
        cat_cat_diff_df = cat_cat_diff_df.sort_values(
            by="Difference of Mean Response", ascending=False
        )
        cat_cat_diff_styler = cat_cat_diff_df.style.format(
            {
                "Bin Plot": make_clickable,
                "Residual Plot": make_clickable,
            }
        )
        cat_cat_diff_styler = cat_cat_diff_styler.to_html()

    # correlation plots
    corr_df = pd.DataFrame(
        columns=[
            "Pair Type",
            "Correlation Plot",
        ]
    )
    # Continuous / Continuous
    con_pt = "Continuous_Continuous"
    con_corr = df[con_pred].corr()
    con_corr_plot = corr_plot(con_corr, con_pt)
    new_row_con = {
        "Pair Type": con_pt,
        "Correlation Plot": con_corr_plot,
    }
    corr_df = corr_df.append(new_row_con, ignore_index=True)
    # Continuous / Categorical
    con_cat_pt = "Continuous_Categorical"
    con_cat_corr_plot = corr_plot(con_cat_corr_d, con_cat_pt)
    new_row_con_cat = {
        "Pair Type": con_cat_pt,
        "Correlation Plot": con_cat_corr_plot,
    }
    corr_df = corr_df.append(new_row_con_cat, ignore_index=True)
    # Categorical / Categorical
    cat_pt = "Categorical_Categorical"
    cat_corr_plot = corr_plot(cat_cat_corr_df, cat_pt)
    new_row_cat = {
        "Pair Type": cat_pt,
        "Correlation Plot": cat_corr_plot,
    }
    corr_df = corr_df.append(new_row_cat, ignore_index=True)
    corr_df_html = corr_df.style.format(
        {
            "Correlation Plot": make_clickable,
        }
    ).to_html()

    hw4_df = pd.DataFrame(
        columns=[
            "Response",
            "Predictor",
            "Histogram",
            "Violin",
            "t-value",
            "p-value",
            "Log Reg Plot",
            "MWR - Unweighted",
            "MWR - Weighted",
            "MWR Plot",
        ]
    )
    hw4_styler = ""
    for column in pred_cols:
        pred = df[pred_cols]
        resp = df[resp_col]
        pn = pred[column].name
        p_type = "continuous"
        # print(f"Predictor {column} is {p_type}.")
        pnt = f"{column} ({p_type})"
        # cont_pred_list.append(pred[column].name)
        # plots for boolean response with continuous predictor
        # was not able to get distribution plot work, using histogram with run instead
        df_cont = pred[column].values
        fig = px.histogram(
            df_cont,
            color=resp,
            marginal="rug",
            title=f"Predictor {pred[column].name} vs. Response {resp.name}",
        )
        # fig.show()
        fig_html = "output/" + f"{pn}_hist_plot.html"
        fig.write_html(fig_html)

        fig1 = px.violin(
            df_cont,
            color=resp,
        )
        # fig1.show()

        fig1_html = "output/" + f"{pn}_violin_plot.html"
        fig1.write_html(fig1_html)

        # logistic_regression
        feature_name = pred[column].name
        p = statsmodels.api.add_constant(pred[column].values)
        log_reg = statsmodels.api.Logit(resp.values, p)
        log_reg_fit = log_reg.fit()
        t_val = round(log_reg_fit.tvalues[1], 6)
        # p_val = "{:.6e}".format(log_reg_fit.pvalues[1])
        p_val = log_reg_fit.pvalues[1]
        # Plot the figure
        fig_lr = px.scatter(x=pred[column], y=resp, trendline="lowess")
        fig_lr.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_val}) (p-value={p_val})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        # fig_lr.show()
        fig_log_html = "output/" + f"{pn}_log_reg_plot.html"
        fig_lr.write_html(fig_log_html)
        # plot_lr_list.append(fig1_html)

        # Difference with Mean of Response
        mean, edges, bin_number = stats.binned_statistic(
            df[column], df[resp_col], statistic="mean", bins=10
        )
        count, edges, bin_number = stats.binned_statistic(
            df[column], df[resp_col], statistic="count", bins=10
        )
        pop_mean = np.mean(df[resp_col])
        edge_centers = (edges[:-1] + edges[1:]) / 2
        mean_diff = mean - pop_mean
        mdsq = mean_diff**2
        pop_prop = count / len(df[resp_col])
        wmdsq = pop_prop * mdsq
        msd = np.nansum(mdsq) / 10
        wmsd = np.nansum(wmdsq) / 10
        pop_mean_list = [pop_mean] * 10

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=edge_centers, y=count, name="Population"), secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=edge_centers,
                y=mean,
                name="Mi - Mpop",
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=edge_centers,
                y=pop_mean_list,
                name="Mpop",
            ),
            secondary_y=True,
        )

        fig_mwr_html = "output/" + f"{pn}_mwr_plot.html"
        fig.write_html(fig_mwr_html)

        new_row = {
            "Response": resp_col,
            "Predictor": pnt,
            "Histogram": fig_html,
            "Violin": fig1_html,
            "t-value": t_val,
            "p-value": p_val,
            "Log Reg Plot": fig_log_html,
            "MWR - Unweighted": msd,
            "MWR - Weighted": wmsd,
            "MWR Plot": fig_mwr_html,
        }
        hw4_df = hw4_df.append(new_row, ignore_index=True)
        hw4_styler = hw4_df.style.format(
            {
                "Histogram": make_clickable,
                "Violin": make_clickable,
                "Log Reg Plot": make_clickable,
                "MWR Plot": make_clickable,
            }
        ).to_html()

    # creating HTML report
    file = open("report.html", "w")
    # Correlation
    file.write(corr_df_html)
    # Continuous / Continuous
    file.write(con_con_df_html)
    file.write(con_con_diff_styler)
    # Continuous / Categorical
    file.write(full_pred_df_html)
    file.write(con_cat_corr_df_html)
    file.write(con_cat_diff_html)
    # Categorical / Categorical
    file.write(cat_cat_html_styler)
    file.write(cat_cat_diff_styler)
    # HW4
    file.write(hw4_styler)
    file.close()

    filename = "file:///" + os.getcwd() + "/report.html"
    webbrowser.open_new_tab(filename)
