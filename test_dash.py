import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, curdoc
from bokeh.driving import linear
import torch

# Create initial data
x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))

# Create a new plot
plot = figure(height=400, width=400, tools="",
              title="Real-time sine wave",
              x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])

# Add a line to the plot
plot.line('x', 'y', source=source, line_width=2, alpha=0.85)

# Define an update function
@linear(m=0.05, b=0)  # m defines the step of phase shift
def update(step):
    # Compute the new y values
    y = np.sin(x + step)
    source.data = dict(x=x, y=y)

# Add the update function to curdoc
curdoc().add_root(plot)
curdoc().add_periodic_callback(update, 10)  # Update every 100 milliseconds

# To start the visualization, run `bokeh serve real_time_bokeh.py` in the terminal

