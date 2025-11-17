import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import util.math as mu

#generate random points
def get_random_points():
    alphas = 2 * np.pi * np.random.rand(4)
    r_values = np.random.rand(4)

    x_values = r_values * np.cos(alphas)
    y_values = r_values * np.sin(alphas)

    points = np.column_stack((x_values, y_values))
    return points

if __name__ == '__main__':
    for i in tqdm(range(10)):
        points = get_random_points()
        coeffs = mu.generate_hyperbola(points)
        xs, ys1, ys2 = mu.generate_conic_linspaces(coeffs, n=2000, x_min=-2, x_max=2)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=xs, y=ys1, mode="lines", 
            line=dict(color='dodgerblue')
        ))
        
        fig.add_trace(go.Scatter(
            x=xs, y=ys2, mode="lines", 
            line=dict(color='dodgerblue'), 
            showlegend=False # same label, hide second entry
        ))
            
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode="markers",
            marker=dict(size=10, color="crimson", symbol="circle"),
            name="Given points"
        ))
            
        fig.update_layout(
            title="Rectangular hyperbola",
            xaxis_title="x", yaxis_title="y",
            xaxis=dict(scaleanchor="y", scaleratio=1, range=[-2, 2]),
            yaxis=dict(range=[-2,2]),
            showlegend=True,
            template="plotly_white"
        )

        fig.write_image(f'plots/random/{i}.png')