import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Source for stats.binned_statistic:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
# https://stackoverflow.com/questions/51828828/given-the-scipy-stats-binned-statistic-function-how-to-work-with-diferent-si
# Plot Reference: https://stackoverflow.com/questions/62122015/how-to-add-traces-in-plotly-express


# Continuous
def con_diff(df, predictor, response):
    mean, edges, bin_number = stats.binned_statistic(
        df[predictor], df[response], statistic="mean", bins=10
    )
    count, edges, bin_number = stats.binned_statistic(
        df[predictor], df[response], statistic="count", bins=10
    )
    pop_mean = np.mean(df[response])
    edge_centers = (edges[:-1] + edges[1:]) / 2
    mean_diff = mean - pop_mean
    mdsq = mean_diff**2
    pop_prop = count / len(df[response])
    wmdsq = pop_prop * mdsq
    msd = np.nansum(mdsq) / 10
    wmsd = np.sum(wmdsq)
    pop_mean_list = [pop_mean] * 10

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=edge_centers,
            y=mean_diff,
            name="Mi - Mpop",
        ),
        secondary_y=False,
    )

    fig.add_trace(go.Bar(x=edge_centers, y=count, name="Population"), secondary_y=True)
    fig.add_trace(
        go.Scatter(
            y=pop_mean_list,
            x=edge_centers,
            name="Mpop",
        )
    )

    con_diff_plot = "output/" + f"{predictor}_{response}_con_diff.html"

    fig.write_html(
        file=con_diff_plot,
        include_plotlyjs="cdn",
    )

    return con_diff_plot, msd, wmsd


# Categorical
def cat_diff(df, predictor, response):
    categories = df[predictor].unique().astype(str)
    categories = sorted(categories)
    mini_df = df[[predictor, response]]
    mean = mini_df.groupby([predictor]).mean()
    count = mini_df.groupby([predictor]).count()
    pop_mean = np.mean(df[response])
    mean_diff = mean.values - pop_mean
    mdsq = mean_diff**2
    pop_prop = count.values / len(df[response])
    wmdsq = pop_prop * mdsq
    msd = np.nansum(mdsq) / len(categories)
    wmsd = np.nansum(wmdsq)

    pop_mean_list = [pop_mean] * len(categories)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=categories,
            y=mean_diff.flatten(),
            name="Mi - Mpop",
            mode="lines+markers",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(x=categories, y=count.values, name="Population"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            y=pop_mean_list,
            x=categories,
            name="Mpop",
        )
    )

    cat_diff_plot = "output/" + f"{predictor}_{response}_cat_diff.html"

    fig.write_html(
        file=cat_diff_plot,
        include_plotlyjs="cdn",
    )

    return cat_diff_plot, msd, wmsd


# Continuous / Continuous
def con_con_diff(df, pred1, pred2, response):
    pred1_edges = np.histogram_bin_edges(df[pred1], bins=10)
    pred2_edges = np.histogram_bin_edges(df[pred2], bins=10)

    bin_mean, pred1_edges, pred2_edges, b = stats.binned_statistic_2d(
        df[pred1],
        df[pred2],
        df[response],
        statistic="mean",
        bins=(pred1_edges, pred2_edges),
    )

    bin_count, pred1_edges, pred2_edges, b = stats.binned_statistic_2d(
        df[pred1],
        df[pred2],
        df[response],
        statistic="count",
        bins=(pred1_edges, pred2_edges),
    )

    pop_mean = df[response].mean()

    pred1_centers = (pred1_edges[:-1] + pred1_edges[1:]) / 2
    pred2_centers = (pred2_edges[:-1] + pred2_edges[1:]) / 2
    pred1_centers = pred1_centers.tolist()
    pred2_centers = pred2_centers.tolist()

    diff = bin_mean - pop_mean
    sq_diff = np.square(diff)
    sum_sq_diff = np.nansum(sq_diff)
    total_bin = bin_mean.size
    msd = sum_sq_diff / total_bin
    pop_prop = bin_count / len(df[response])
    w_sq_diff = pop_prop * sq_diff
    wmsd = np.nansum(w_sq_diff)

    fig = ff.create_annotated_heatmap(
        x=pred2_centers,
        y=pred1_centers,
        z=bin_mean,
        annotation_text=bin_mean,
        colorscale="curl",
        showscale=True,
        customdata=pop_prop,
    )

    fig.update_layout(
        title_text=f"{pred1} & {pred2}",
    )
    filename1 = "output/" + f"{pred1}_{pred2}_bin.html"

    fig.write_html(
        file=filename1,
        include_plotlyjs="cdn",
    )

    # residual plot
    fig_2 = ff.create_annotated_heatmap(
        x=pred2_centers,
        y=pred1_centers,
        z=diff,
        colorscale="curl",
    )

    fig_2.update_layout(
        title_text=f"{pred1} & {pred2} Bin Average Residual",
        xaxis=dict(tickmode="array", tickvals=pred2_edges),
        yaxis=dict(tickmode="array", tickvals=pred1_edges),
    )
    filename2 = "output/" + f"{pred1}_{pred2}_residual.html"

    fig_2.write_html(
        file=filename2,
        include_plotlyjs="cdn",
    )

    return filename1, filename2, msd, wmsd


# Continuous / Categorical
def con_cat_diff(df, pred1, pred2, response):
    pred1_edges = np.histogram_bin_edges(df[pred1], bins="sturges")
    categories = df[pred2].unique()
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit_transform(categories)

    # using later for graphs and also for calculations
    pred1_centers = (pred1_edges[:-1] + pred1_edges[1:]) / 2
    pred1_centers = pred1_centers.tolist()
    all_bin = pd.DataFrame(index=int_encoded, columns=pred1_centers)
    bin_count = pd.DataFrame(index=int_encoded, columns=pred1_centers)
    bin_mean = pd.DataFrame(index=int_encoded, columns=pred1_centers)

    for x in range(len(categories)):
        for i in range(len(pred1_edges) - 1):

            temp_bin = df[
                (df[pred2] == categories[x])
                & (df[pred1] >= pred1_edges[i])
                & (df[pred1] < pred1_edges[i + 1])
            ][response]
            all_bin.at[int_encoded[x], pred1_centers[i]] = temp_bin
            bin_count.loc[int_encoded[x], pred1_centers[i]] = len(temp_bin)
            bin_mean.loc[int_encoded[x], pred1_centers[i]] = np.mean(temp_bin)

    pop_mean = df[response].mean()

    diff = bin_mean - pop_mean
    sq_diff = np.square(diff)
    sum_sq_diff = np.nansum(sq_diff)
    total_bin = len(pred1_centers) * len(categories)
    msd = sum_sq_diff / total_bin
    pop_prop = bin_count / len(df[response])
    w_sq_diff = pop_prop * sq_diff
    wmsd = np.nansum(w_sq_diff)

    fig = ff.create_annotated_heatmap(
        x=pred1_centers,
        y=categories.tolist(),
        z=bin_mean.values.tolist(),
    )

    fig.update_layout(
        title_text=f"{pred1} & {pred2} Bin Mean of Response",
    )
    plot_diff_bin = "output/" + f"{pred1}_{pred2}_diff_bin.html"

    fig.write_html(
        file=plot_diff_bin,
        include_plotlyjs="cdn",
    )

    fig_2 = ff.create_annotated_heatmap(
        x=pred1_centers,
        y=categories.tolist(),
        z=diff.values.tolist(),
    )

    fig_2.update_layout(
        title_text=f"{pred1} & {pred2} Bin Mean",
    )
    plot_residual = "output/" + f"{pred1}_{pred2}_diff_residual.html"

    fig_2.write_html(
        file=plot_residual,
        include_plotlyjs="cdn",
    )

    return plot_diff_bin, plot_residual, msd, wmsd


# Categorical / Categorical
def cat_cat_diff(df, pred1, pred2, response):
    cat1 = df[pred1].unique().astype(str)
    cat2 = df[pred2].unique().astype(str)
    cat1 = sorted(cat1)
    cat2 = sorted(cat2)

    # get mean and bin count
    bin_mean = pd.crosstab(
        index=df[pred1],
        columns=df[pred2],
        values=df[response],
        aggfunc="mean",
        margins=False,
    )
    pop_mean = df[response].mean()
    bin_count = pd.crosstab(index=df[pred1], columns=df[pred2], margins=False)

    diff = bin_mean - pop_mean
    sq_diff = np.square(diff)
    sum_sq_diff = np.nansum(sq_diff)
    total_bin = len(cat1) * len(cat2)
    msd = sum_sq_diff / total_bin
    pop_prop = bin_count / len(df[response])
    w_sq_diff = pop_prop * sq_diff
    wmsd = np.nansum(w_sq_diff)

    pop_prop = pop_prop.round(3)
    fig = ff.create_annotated_heatmap(
        x=bin_mean.columns.tolist(),
        y=bin_mean.index.tolist(),
        z=bin_mean.values,
    )

    fig.update_layout(title_text=f"{pred1} & {pred2} Bin Averages of Response")
    file_hm = "output/" + f"{pred1}_{pred2}_diff_heatmap.html"

    fig.write_html(
        file=file_hm,
        include_plotlyjs="cdn",
    )

    # residual plot
    fig_2 = ff.create_annotated_heatmap(
        x=cat2,
        y=cat1,
        z=diff.values,
        colorscale="curl",
        customdata=pop_prop,
        annotation_text=diff.values.round(3),
        hoverongaps=False,
        showscale=True,
        zmid=pop_mean,
        zmin=df[response].min(),
        zmax=df[response].max(),
    )

    fig_2.update_layout(title_text=f"{pred1} & {pred2} Bin Average")
    file_res = "output/" + f"{pred1}_{pred2}_residual.html"

    fig_2.write_html(
        file=file_res,
        include_plotlyjs="cdn",
    )

    return file_hm, file_res, msd, wmsd
