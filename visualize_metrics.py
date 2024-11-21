import plotly.graph_objects as go

from constants import ALL_METRICS_RESULTS, HETEROPHILY_METRICS

SCALER = 120


def metrics_visualization(graph_type: str, lvl: float, calculated_metrics: dict):
    hms = [hm for hm, _ in HETEROPHILY_METRICS.items() if hm in calculated_metrics]
    metrics = [(calculated_metrics[hm] * SCALER, hm) for hm in hms]
    metrics.sort()
    idx_map = {hm[1]: idx for idx, hm in enumerate(metrics)}

    colors = [idx_map[hm] for hm in hms]

    fig = go.Figure(data=[go.Scatter(
        y=[HETEROPHILY_METRICS[hm] for hm in hms],
        x=[max(0, calculated_metrics[hm]) * SCALER for hm in hms],
        customdata=[calculated_metrics[hm] for hm in hms],
        mode='markers',
        marker=dict(
            size=[max(0.05, calculated_metrics[hm]) * SCALER for hm in hms],
            showscale=True,
            color=colors,
            colorbar=dict(
                title=dict(text="Homophily Level")
            ),
            colorscale="Viridis"
        )
    )])
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_traces(
        hovertemplate=
        "<b>%{y}</b><br>" +
        "%{customdata:.5f}<br>" +
        "<extra></extra>",
    )
    fig.update_layout(title={
        'text': f"Comparing Metrics for Measuring Homophily in a {graph_type} Graph ({lvl})",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        height=len(hms) * 30 + 200,
    )

    return fig


def homophily_lvl_visulization(graph_type: str, metric_name: str):
    datapoints = ALL_METRICS_RESULTS[graph_type][metric_name]
    reverse_map = {
        "Kernel Linear Regression": "kernel_reg0",
        "Kernel Non-linear Regression": "kernel_reg1",
        "Gaussian Naive Bayes": "gnb"
    }
    if metric_name in reverse_map:
        datapoints = ALL_METRICS_RESULTS[graph_type][reverse_map[metric_name]]
    else:
        datapoints = ALL_METRICS_RESULTS[graph_type][metric_name]

    kw = "Beta" if graph_type == "GenCat" else "Homophily Level"
    graph_type = "Regular" if graph_type == "RG" else graph_type

    fig = go.Figure()
    for homophily_level, pts in datapoints.items():
        fig.add_trace(go.Box(y=pts, quartilemethod="linear", name=homophily_level, boxpoints='all',
                             jitter=0.5,
                             whiskerwidth=0.2, ))
    fig.update_traces(jitter=0)
    fig.update_layout(
        title={'text': f"Evaluation of {metric_name} for {graph_type} Synthetic Graphs at Different {kw}s",
               'y': 0.9,
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top'},
        xaxis={"title": {"text": kw}},
        yaxis={"title": {"text": metric_name}},
        showlegend=False,
    )
    return fig
