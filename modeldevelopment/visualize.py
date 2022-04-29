import logging

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def plot_acc_spd_values(run_name, x_axis_threshold, threshold_values, accuracy_values, spd_values):
    try:

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=threshold_values,
                       y=accuracy_values,
                       name="Accuracy Values"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=threshold_values,
                       y=spd_values,
                       name="SPD Values"),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text=str(run_name).upper()
        )

        # Set x-axis title
        fig.update_xaxes(title_text=f"<b>{x_axis_threshold}</b> values")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Accuracy</b> values",
                         secondary_y=False)
        fig.update_yaxes(title_text="<b>SPD</b> values", secondary_y=True)

        fig.write_image(f"./reports/figures/{run_name}.png")

    except Exception as e:
        logger.error(e)
