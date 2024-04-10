import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def visualize_seasonal_decompose(
    series: pd.Series, title_prefix: str = "", period: int = 24
) -> None:
    """
    Perform and plot seasonal decomposition of the series using Plotly to assist in identifying seasonality.

    Args:
        series (pd.Series): Time series data.
        title_prefix (str): Optional prefix for the plot title to indicate the context.
        period (int): The period of the seasonality.
    """
    decomposition = seasonal_decompose(series, model="additive", period=period)

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"{title_prefix}Observed",
            f"{title_prefix}Trend",
            f"{title_prefix}Seasonal",
            f"{title_prefix}Residual",
        ),
    )

    # Plot the observed data
    fig.add_trace(
        go.Scatter(
            x=series.index, y=decomposition.observed, mode="lines", name="Observed"
        ),
        row=1,
        col=1,
    )

    # Plot the trend component
    fig.add_trace(
        go.Scatter(x=series.index, y=decomposition.trend, mode="lines", name="Trend"),
        row=2,
        col=1,
    )

    # Plot the seasonal component
    fig.add_trace(
        go.Scatter(
            x=series.index, y=decomposition.seasonal, mode="lines", name="Seasonal"
        ),
        row=3,
        col=1,
    )

    # Plot the residual component
    fig.add_trace(
        go.Scatter(
            x=series.index, y=decomposition.resid, mode="lines", name="Residual"
        ),
        row=4,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"{title_prefix}Seasonal Decomposition",
        template="plotly_white",
    )

    fig.show()


def plot_forecast_vs_actual(
    series: pd.Series,
    forecast_mean: pd.Series,
    forecast_conf_int: pd.DataFrame,
    title_prefix: str = "",
):
    """
    Plot actual data vs forecasted data with confidence intervals using Plotly.

    Args:
        series (pd.Series): Actual time series data.
        forecast_mean (pd.Series): Forecasted mean values.
        forecast_conf_int (pd.DataFrame): Forecasted confidence intervals.
        title_prefix (str): Optional prefix for the plot title to provide context.
    """
    # Creating a Plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Adding actual data series
    fig.add_trace(
        go.Scatter(x=series.index, y=series, mode="lines", name="Actual Data"),
        secondary_y=False,
    )

    # Adding forecast mean series
    fig.add_trace(
        go.Scatter(
            x=forecast_mean.index, y=forecast_mean, mode="lines", name="Forecast"
        ),
        secondary_y=False,
    )

    # Adding forecast confidence interval area
    fig.add_trace(
        go.Scatter(
            x=forecast_conf_int.index.tolist() + forecast_conf_int.index[::-1].tolist(),
            y=forecast_conf_int.iloc[:, 0].tolist()
            + forecast_conf_int.iloc[:, 1][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
        ),
        secondary_y=False,
    )

    # Setting plot titles and labels
    fig.update_layout(
        title=f"{title_prefix}Actual vs Forecasted Data",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
        template="plotly_white",
    )

    fig.show()


def visualize_data_analysis(series: pd.Series, title_prefix: str = ""):
    """
    Perform comprehensive data visualizations, including raw data plotting,
    autocorrelation, partial autocorrelation, and seasonal decomposition.

    Args:
        series (pd.Series): Time series data with a datetime index.
        title_prefix (str): Optional prefix for the plot titles to provide context.
    """
    # Plot the raw data
    plt.figure(figsize=(12, 4))
    plt.plot(series, label=f"{title_prefix}Raw Data")
    plt.title(f"{title_prefix}Time Series Data")
    plt.legend()
    plt.show()

    # ACF and PACF plots
    plot_acf_pacf(series, title_prefix)

    # Seasonal decomposition
    visualize_seasonal_decompose(series, title_prefix)


def visualize_boxplots(
    series: pd.Series, title: str = "Boxplot of Time Series Data", frequency: str = "H"
):
    """
    Generate box plots or scatter plots of the time series data for specified frequency: Hourly, Daily, Weekly, or Monthly.

    Args:
        series (pd.Series): Time series data with a datetime index.
        title (str): Title for the plot.
        frequency (str): Frequency for aggregating data. 'H' for hourly, 'D' for daily, 'W' for weekly, 'M' for monthly.
    """
    df_for_boxplot = series.copy().to_frame()
    df_for_boxplot["Year"] = df_for_boxplot.index.year
    df_for_boxplot["Month"] = df_for_boxplot.index.month
    df_for_boxplot["Day"] = df_for_boxplot.index.day
    df_for_boxplot["Hour"] = df_for_boxplot.index.hour
    if frequency == "W":
        df_for_boxplot["Week"] = df_for_boxplot.index.isocalendar().week

    # Map frequency to plot data and labels
    freq_to_label = {"M": "Month", "W": "Week", "D": "Day", "H": "Hour"}
    x_label = freq_to_label.get(frequency, "Time")

    plt.figure(figsize=(12, 6))
    if frequency in ["D", "H"]:  # Use scatter plot for dense data points
        sns.stripplot(
            data=df_for_boxplot,
            x=freq_to_label[frequency],
            y=series.name,
            jitter=0.25,
            size=2.5,
        )
    else:  # Use boxplot for less dense data points
        sns.boxplot(data=df_for_boxplot, x=freq_to_label[frequency], y=series.name)
    plt.title(title)
    plt.xlabel(x_label)
    plt.xticks(rotation=45)  # Helps with readability, especially for large datasets
    plt.grid(True)
    plt.show()


def plot_acf_pacf(series: pd.Series, title_prefix: str = ""):
    """
    Plot Autocorrelation and Partial Autocorrelation.

    Args:
        series (pd.Series): Time series data or model residuals.
        title_prefix (str): Optional prefix for the plot titles to indicate if these are raw data or residuals.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_acf(series, ax=axes[0])
    plot_pacf(series, ax=axes[1])
    axes[0].set_title(f"{title_prefix}Autocorrelation")
    axes[1].set_title(f"{title_prefix}Partial Autocorrelation")
    plt.tight_layout()
    plt.show()
