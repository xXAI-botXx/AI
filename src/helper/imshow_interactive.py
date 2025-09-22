"""
Installation:

pip install plotly
pip install "notebook>=5.3" "ipywidgets>=7.2"
pip install jupyterlab "ipywidgets>=7.5"
jupyter labextension install jupyterlab-plotly

Usage:
```python
import sys
sys.path += ["../AI/src/helper"]

from imshow_interactive import interactive_line_plot, interactive_image

# Line plot
import numpy as np
data = [np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100))]
interactive_line_plot(data, labels=["sin(x)", "cos(x)"])

# Image
import cv2
img = cv2.imread("example.png", cv2.IMREAD_GRAYSCALE)
interactive_image(img, title="Grayscale Image")
```
"""

# plotly_helpers.py
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def interactive_line_plot(data, x=None, labels=None, title="Interactive Line Plot"):
    """
    Create an interactive line plot with Plotly.

    Parameters
    ----------
    data : list or np.ndarray
        List of 1D arrays or 2D array where each row is a line.
    x : array-like, optional
        X-axis values. If None, uses np.arange for each line.
    labels : list of str, optional
        Names for each line.
    title : str, optional
        Plot title.
    """
    fig = go.Figure()

    if isinstance(data, np.ndarray) and data.ndim == 2:
        n_lines = data.shape[0]
        data_list = [data[i] for i in range(n_lines)]
    elif isinstance(data, list):
        data_list = data
    else:
        raise ValueError("Data must be a list of 1D arrays or a 2D numpy array.")

    for i, line in enumerate(data_list):
        x_vals = x if x is not None else np.arange(len(line))
        name = labels[i] if labels and i < len(labels) else f"Line {i+1}"
        fig.add_trace(go.Scatter(x=x_vals, y=line, mode='lines+markers', name=name))

    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y", template="plotly_white")
    fig.show()


def interactive_image(img, title="Interactive Image", colormap="gray"):
    """
    Display a 2D image interactively using Plotly.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale or 3D RGB image.
    title : str
        Title of the plot.
    colormap : str
        Color scale to use for grayscale images. Ignored for RGB images.
    """
    img = np.asarray(img)

    if img.ndim == 2:  # grayscale
        fig = px.imshow(img, color_continuous_scale=colormap, origin='upper')
    elif img.ndim == 3 and img.shape[2] in [3, 4]:  # RGB or RGBA
        fig = px.imshow(img, origin='upper')
    else:
        raise ValueError("Image must be 2D grayscale or 3D RGB(A).")

    fig.update_layout(title=title, xaxis_showgrid=False, yaxis_showgrid=False)
    fig.show()


def interactive_images_grid(images, titles=None, cols=3, colormap="gray"):
    """
    Display multiple images in a grid interactively.

    Parameters
    ----------
    images : list of np.ndarray
        List of 2D or 3D images.
    titles : list of str, optional
        Titles for each image.
    cols : int
        Number of columns in the grid.
    colormap : str
        Colormap for grayscale images.
    """
    from math import ceil

    n_images = len(images)
    rows = ceil(n_images / cols)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles if titles else [f"Image {i+1}" for i in range(n_images)])

    for i, img in enumerate(images):
        r = i // cols + 1
        c = i % cols + 1
        if img.ndim == 2:
            fig.add_trace(go.Image(z=img), row=r, col=c)
        elif img.ndim == 3:
            fig.add_trace(go.Image(z=img), row=r, col=c)
        else:
            raise ValueError("Each image must be 2D grayscale or 3D RGB(A).")

    fig.update_layout(height=300*rows, width=300*cols, showlegend=False, title_text="Image Grid")
    fig.show()


# Optional helper for subplots
from plotly.subplots import make_subplots



