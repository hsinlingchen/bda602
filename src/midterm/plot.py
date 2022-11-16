import pandas as pd
import plotly.graph_objects as go
from plotly import express as px

# I was not able to get the distribution plot to work on my laptop (same as HW4),
# here continues using historgram + rug as substitute.


def corr_plot(df, pt):
    fig = px.imshow(df)
    html = "output/" + f"{pt}_corrmap.html"
    fig.write_html(html)
    return html


def heatmap(df, predictor, response):
    crosstab_df = pd.crosstab(index=df[predictor], columns=df[response], margins=False)

    fig = go.Figure(
        data=go.Heatmap(
            x=crosstab_df.index,
            y=crosstab_df.columns,
            z=crosstab_df.values,
            zmin=0,
            zmax=crosstab_df.max().max(),
        )
    )

    fig.update_layout(
        title=f"{predictor} {response} Heatmap",
        xaxis_title=f"{predictor}",
        yaxis_title=f"{response}",
    )
    html = "output/" + f"{predictor}_{response}_heatmap.html"
    fig.write_html(html)
    return html


def con_cat_plot(df, predictor, response):
    con_cat_hist = "output/" + f"{predictor}_{response}_hist_plot.html"
    con_cat_vio = "output/" + f"{predictor}_{response}_violin_plot.html"
    group_labels = df[response].unique()
    hist_data = []

    for label in group_labels:
        sub_df = df[df[response] == label]
        group = sub_df[predictor]
        hist_data.append(group)

    # Using Histogram + Rug as a substitute to distribution
    fig_1 = px.histogram(df, x=predictor, color=response, marginal="rug")
    fig_1.update_layout(
        title=f"{predictor} by {response} distribution plot",
        xaxis_title=predictor,
        yaxis_title=response,
    )

    fig_1.write_html(
        file=con_cat_hist,
        include_plotlyjs="cdn",
    )

    fig_2 = px.violin(df, y=predictor, color=response, box=True)

    fig_2.update_layout(
        title=f"{response} by {predictor} violin plot",
        xaxis_title=predictor,
        yaxis_title=response,
    )

    fig_2.write_html(
        file=con_cat_vio,
        include_plotlyjs="cdn",
    )

    return con_cat_hist, con_cat_vio
