import util.data as du
import util.graphing as gu
import util.math as mu
import numpy as np

import plotly.graph_objects as go
import pandas as pd

def main(val_of_interest):
    # systems = du.a1689_3_miscreants
    systems = du.a1689_quads
    
    graph = gu.Graph(
        title='Abell 1689 Miscreants',
        xaxis_title='X (pixels)',
        yaxis_title='Y (pixels)'
    )
    graph.update_axes(
        xaxis_range=[-2000, 2000],
        yaxis_range=[-1300, 1600]
    )
    
    x_values = []
    y_values = []
    oreintations = []
    names = []
    
    
    for i, system in enumerate(systems):
        images = system.images - 2048  # Transform images by (-2048, -2048)
        
        avg_point = (np.mean(images, axis=0))
        avg_x = avg_point[0]
        avg_y = avg_point[1]
        
        asymptotes = mu.get_asymptotes(mu.generate_hyperbola(images))
        ms = np.array([asymptotes[0][0], asymptotes[1][0]])
        
        if np.abs(ms[0]) > np.abs(ms[1]):
            m = ms[0]
        else:
            m = ms[1]
        
        orientation = np.arctan(m)
        
        if orientation < 0:
            orientation += np.pi
        
        x_values.append(avg_x)
        y_values.append(avg_y)
        oreintations.append(orientation)
        names.append(system.name)
        
        # print(f'System: {system.name}, Avg Point: {avg_point}, Orientation: {orientation:.2f} radians, Slope: {asymptotes[0][0]:.2f}')
        
        graph.add_trace(
            images[:, 0],
            images[:, 1],
            'images',
            color_idx=i,
            name=f'Images {system.name}'
        )
        
        graph.add_trace(
            avg_point[0],
            avg_point[1],
            'source',
            color_idx=i,
            name=f'Source {system.name}'
        )
        
    # graph.show()
    
    fig = go.Figure()
    
    df = pd.DataFrame({
        'X': x_values,
        'Y': y_values,
        'Orientation': oreintations,
        'Names': names
    })
    
    coefficients = np.polyfit(df[val_of_interest], df['Orientation'], 1)
    polynomial_function = np.poly1d(coefficients)
    y_values_best_fit = polynomial_function(df[val_of_interest])
    
    # calculate r^2
    residuals = df['Orientation'] - y_values_best_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((df['Orientation'] - np.mean(df['Orientation']))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f'Best fit coefficients: {coefficients}')
    print(f'R-squared: {r_squared:.4f}')
    
    fig.add_trace(go.Scatter(
        x=df[val_of_interest],
        y=df['Orientation'],
        mode='markers+text',
        text=df['Names'],
        textposition='top center',
        marker=dict(size=10, color='blue'),
        name='Orientation'
    ))
    
    fig.add_trace(go.Scatter(
        x=df[val_of_interest],
        y=y_values_best_fit,
        mode='lines',
        line=dict(color='red', width=2),
        name='Best Fit Line'
    ))
    
    fig.update_layout(
        title=f'Orientation vs {val_of_interest} (R² = {r_squared:.4f})',
        xaxis_title=f'{val_of_interest} (pixels)',
        yaxis_title='Orientation (radians)',
        showlegend=True
    )
    fig.show()
    
    
    
if __name__ == "__main__":
    val_of_interest = 'X'
    
    main(val_of_interest)