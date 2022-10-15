import sys

import pandas as pd
import plotly.graph_objects as go
import statsmodels.api
from plotly import express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor


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

    r_type_list = []
    p_type_list = []
    plot_h_list = []
    plot_hist_list = []
    plot_v_list = []
    plot_lr_list = []
    t_list = []
    p_list = []
    rf_imp_list = []
    umwr_list = []
    mwr_list = []

    # determine if the response is continuous, boolean, or categorical
    if float(resp.nunique()) / resp.count() < 0.05:
        r_type = "boolean"
        print(f"Response is {r_type}.")
        r_type_list.append(f"{resp.name} ({r_type})")
    elif resp is str:
        r_type = "categorical"
        print(f"Response is {r_type}.")
        r_type_list.append(f"{resp.name} ({r_type})")
    else:
        r_type = "continuous"
        print(f"Response is {r_type}.")
        r_type_list.append(f"{resp.name} ({r_type})")

    # determine if the predictors are continuous, boolean, or categorical
    for column in pred:
        if pred[column].dtype.name in ["object", "int64"]:
            p_type = "categorical"
            print(f"Predictor {column} is {p_type}.")
            p_type_list.append(f"{column} ({p_type})")
            # plots for boolean response with categorical predictor
            pn = pred[column].name
            rn = resp.name
            df = pd.DataFrame({pn: pred[column], rn: resp})
            fig_h = px.density_heatmap(
                df,
                x=pn,
                y=rn,
                title=f"Predictor {pn} vs. Response {rn}",
            )
            # fig_h.show()
            fig_html = f"{pn}_heatmap_plot.html"
            fig_h.write_html(fig_html)
            plot_h_list.append(fig_html)
            plot_hist_list.append("NA")
            plot_v_list.append("NA")
            t_val = "NA"
            t_list.append(t_val)
            p_val = "NA"
            p_list.append(p_val)
            plot_lr_list.append("NA")

            # Difference with Mean of Response - Category
            predictor = f"{pred[column].name}"
            response = f"{resp.name}"
            target = resp.unique()[0]
            diff_cat_prop = df[resp == target].count() / resp.count()
            diff_cat_dt = pd.DataFrame()
            diff_cat_dt["CategoryCounts"] = df.groupby(by=predictor)[response].count()
            diff_cat_dt["TargetCounts"] = (
                df[resp == target].groupby(by=predictor)[response].count()
            )
            diff_cat_dt["Proportion"] = (
                diff_cat_dt["TargetCounts"] / diff_cat_dt["CategoryCounts"]
            )
            diff_cat_dt["PopProp"] = diff_cat_prop[0]
            diff_cat_dt["MeanDiff"] = diff_cat_dt["Proportion"] - diff_cat_dt["PopProp"]
            diff_cat_dt["MeanSquaredDiff"] = diff_cat_dt["MeanDiff"] ** 2
            diff_cat_dt["WeightedMeanSquaredDiff"] = (
                diff_cat_dt["Proportion"] * diff_cat_dt["MeanSquaredDiff"]
            )

            # print(diff_cat_dt)

            n = len(pred[column].unique())
            uw_msd_sum = diff_cat_dt["MeanSquaredDiff"].sum()
            w_msd_sum = diff_cat_dt["WeightedMeanSquaredDiff"].sum()
            uw_msd_avg = uw_msd_sum / n
            w_msd_avg = w_msd_sum / n
            print(
                f"Unweighted Difference of Mean: {uw_msd_avg}\n"
                f"Weighted Difference of Mean: {w_msd_avg}"
            )
            # Plot - still working on fixing plots
            """
            trace1 = go.Bar(
                x=pred[column],
                y=diff_cat_dt["CategoryCounts"],
                name="population",
            )

            trace2 = go.Scatter(
                x=diff_cat_dt["MeanDiff"],
                y=pred[column],
                name="Population Mean",
            )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(trace1)
            fig.add_trace(trace2, secondary_y=True)
            #fig.add_trace(trace3, secondary_y=True)
            fig["layout"].update(
                height=600,
                width=800,
                title=f"Difference with Mean of Response with Predictor {predictor}",
            )
            fig.show()
            fig_html = f"{predictor}_diff.html"
            fig.write_html(fig_html)
            """
            # Data for Report
            umwr_list.append(f"{uw_msd_avg} ()")
            mwr_list.append(w_msd_avg)
        elif float(pred[column].nunique()) / pred[column].count() < 0.05:
            p_type = "boolean"
            print(f"Predictor {column} is {p_type}.")
            p_type_list.append(f"{column} ({p_type})")
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
            # fig_h.show()
            fig_html = f"{pn}_heatmap_plot.html"
            fig_h.write_html(fig_html)
            plot_h_list.append(fig_html)
            plot_hist_list.append("NA")
            plot_v_list.append("NA")
            t_val = "NA"
            t_list.append(t_val)
            p_val = "NA"
            p_list.append(p_val)
            plot_lr_list.append("NA")
            # Difference with Mean of Response - Boolean
            predictor = f"{pred[column].name}"
            response = f"{resp.name}"
            target = resp.unique()[0]
            bool_prop = df[resp == target].count() / resp.count()
            bool_dt = pd.DataFrame()
            bool_dt["CategoryCounts"] = df.groupby(by=predictor)[response].count()
            bool_dt["TargetCounts"] = (
                df[resp == target].groupby(by=predictor)[response].count()
            )
            bool_dt["Proportion"] = bool_dt["TargetCounts"] / bool_dt["CategoryCounts"]
            bool_dt["PopProp"] = bool_prop[0]
            bool_dt["MeanDiff"] = bool_dt["Proportion"] - bool_dt["PopProp"]
            bool_dt["MeanSquaredDiff"] = bool_dt["MeanDiff"] ** 2
            bool_dt["WeightedMeanSquaredDiff"] = (
                bool_dt["Proportion"] * bool_dt["MeanSquaredDiff"]
            )

            # print(bool_dt)
            n = len(pred[column].unique())
            uw_msd_sum = bool_dt["MeanSquaredDiff"].sum()
            w_msd_sum = bool_dt["WeightedMeanSquaredDiff"].sum()
            uw_msd_avg = uw_msd_sum / n
            w_msd_avg = w_msd_sum / n
            print(
                f"Unweighted Difference of Mean: {uw_msd_avg}\n"
                f"Weighted Difference of Mean: {w_msd_avg}"
            )

            # Plot - still working on fixing plots
            """
            trace1 = go.Bar(
                x=pred[column],
                y=diff_cat_dt["CategoryCounts"],
                name="population",
            )

            trace2 = go.Scatter(
                x=diff_cat_dt["MeanDiff"],
                y=pred[column],
                name="Population Mean",
            )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(trace1)
            fig.add_trace(trace2, secondary_y=True)
            #fig.add_trace(trace3, secondary_y=True)
            fig["layout"].update(
                height=600,
                width=800,
                title=f"Difference with Mean of Response with Predictor {predictor}",
            )
            fig.show()
            fig_html = f"{predictor}_diff.html"
            fig.write_html(fig_html)
            """
            # Data for Report
            umwr_list.append(f"{uw_msd_avg} ()")
            mwr_list.append(w_msd_avg)
        else:
            pn = pred[column].name
            p_type = "continuous"
            print(f"Predictor {column} is {p_type}.")
            p_type_list.append(f"{column} ({p_type})")
            cont_pred_list.append(pred[column].name)
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

            fig_html = f"{pn}_hist_plot.html"
            fig.write_html(fig_html)
            plot_hist_list.append(fig_html)

            fig1 = px.violin(
                df_cont,
                color=resp,
            )
            # fig1.show()

            fig1_html = f"{pn}_violin_plot.html"
            fig1.write_html(fig1_html)
            plot_v_list.append(fig1_html)
            # heatmap not applied
            plot_h_list.append("NA")

            # logistic_regression
            feature_name = pred[column].name
            p = statsmodels.api.add_constant(pred[column].values)
            log_reg = statsmodels.api.Logit(resp.values, p)
            log_reg_fit = log_reg.fit()
            t_val = round(log_reg_fit.tvalues[1], 6)
            p_val = "{:.6e}".format(log_reg_fit.pvalues[1])
            print(f"t-value: {t_val}\np-value: {p_val}")
            t_list.append(t_val)
            p_list.append(p_val)
            # Plot the figure
            fig_lr = px.scatter(x=pred[column], y=resp, trendline="lowess")
            fig_lr.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_val}) (p-value={p_val})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title="y",
            )
            # fig_lr.show()
            fig1_html = f"{pn}_log_reg_plot.html"
            fig_lr.write_html(fig1_html)
            plot_lr_list.append(fig1_html)

            # Difference with Mean of Response
            diff_df = pd.DataFrame()
            ps = pred[column]
            pred_name = f"{ps.name}"
            rs = resp
            pred_df = ps.to_frame()
            pred_df[resp] = rs.values
            pop_mean = rs.mean()
            n = 10
            intervals = pd.qcut(ps.rank(method="first"), n)
            pred_df["LowerBin"] = pd.Series([i.left for i in intervals])
            pred_df["UpperBin"] = pd.Series([i.right for i in intervals])
            labels = ["LowerBin", "UpperBin"]
            diff_df["BinCenters"] = pred_df.groupby(by=labels).median()[pred_name]
            diff_df["BinCount"] = pred_df.groupby(by=labels).count()[pred_name]
            diff_df["Weight"] = diff_df["BinCount"] / ps.count()
            diff_df["BinMean"] = pred_df.groupby(by=labels).mean()[pred_name]
            diff_df["PopulationMean"] = pop_mean
            msd = "MeanSquaredDiff"
            wmsd = "WeightedMeanSquaredDiff"
            diff_df[msd] = (diff_df["BinMean"] - diff_df["PopulationMean"]) ** 2
            diff_df[wmsd] = (
                diff_df["Weight"]
                * (diff_df["BinMean"] - diff_df["PopulationMean"]) ** 2
            )
            diff_df = diff_df.reset_index()
            uw_msd_sum = diff_df[msd].sum()
            w_msd_sum = diff_df[wmsd].sum()
            uw_msd_avg = uw_msd_sum / n
            w_msd_avg = w_msd_sum / n
            print(
                f"Unweighted Difference of Mean: {uw_msd_avg}\n"
                f"Weighted Difference of Mean: {w_msd_avg}"
            )
            # Plot
            trace1 = go.Bar(
                x=diff_df["BinCenters"],
                y=diff_df["BinCount"],
                name="population",
            )

            trace2 = go.Scatter(
                x=diff_df["BinCenters"],
                y=diff_df["PopulationMean"],
                name="Population Mean",
            )
            diff_df["BinMeanMinusPopulationMean"] = (
                diff_df["BinMean"] - diff_df["PopulationMean"]
            )
            # trace 3 came out weird on the plot, still working on solution
            trace3 = go.Scatter(
                x=diff_df["BinCenters"],
                y=diff_df["BinMeanMinusPopulationMean"],
                name="Bin Mean Minus Population Mean",
            )
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(trace1)
            fig.add_trace(trace2, secondary_y=True)
            fig.add_trace(trace3, secondary_y=True)
            fig["layout"].update(
                height=600,
                width=800,
                title=f"Difference with Mean of Response with Predictor {pred_name}",
            )
            # fig.show()
            fig_html = f"{pn}_mwr_plot.html"
            fig.write_html(fig_html)
            mwr_list.append(w_msd_avg)
            umwr_list.append(f"{uw_msd_avg} ({fig_html})")

    # print(p_type_list)
    # Random Forest Feature Importance Rank
    df_X = pred[cont_pred_list]
    df_y = resp
    rf_c = RandomForestRegressor(max_depth=2, random_state=0)
    rf_c.fit(df_X.values, df_y)
    imp = rf_c.feature_importances_
    print(imp)

    # Table Report
    pred_count = len(p_type_list) - 1
    temp_list = list(r_type_list)
    for i in range(pred_count):
        for element in temp_list:
            r_type_list.append(element)
    # print(r_type_list)
    # set up RF VarImp Column
    pred_count_all = len(p_type_list)
    for i in range(pred_count_all):
        rf_imp_list.append(" ")

    writer = pd.ExcelWriter("report.xlsx", engine="openpyxl")
    wb = writer.book
    report_df = pd.DataFrame(
        {
            "Response": r_type_list,
            "Predictor": p_type_list,
            "Heatmap": plot_h_list,
            "Histogram": plot_hist_list,
            "Violin": plot_v_list,
            "t-value": t_list,
            "p-value": p_list,
            "Log Reg Plot": plot_lr_list,
            "RF VarImp": rf_imp_list,
            "MWR - Unweighted": umwr_list,
            "MWR - Weighted": mwr_list,
        }
    )
    # for RF VarImp column, still need to work on the code to add value to the correct cell automatically
    report_df.set_index(["Predictor"])
    report_df.at[2, "RF VarImp"] = imp[0]
    report_df.at[5, "RF VarImp"] = imp[1]

    report_df.to_excel(writer, index=False)
    wb.save("report.xlsx")
    # print(report_df)


if __name__ == "__main__":
    sys.exit(main())
