import util.data as du
import plotly.graph_objects as go
import numpy as np

def main(systems):
    fig = go.Figure()
    
    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    for system in systems:
        points = np.array(system.images)
        num_points = len(points)
        
        if num_points < 4:
            continue
        
        labels = [''] * num_points
        for i in range(num_points):
            labels[i] = f"{system.name}{letter[i]}"
            
        fig.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers+text',
            marker=dict(size=7),
            text=labels,
            textposition='bottom center',
            name=system.name,
            showlegend=False
        ))
        
    fig.update_layout(
        title='Abell 1689 Systems',
        xaxis_title='X',
        yaxis_title='Y'
    )
    
    fig.show()
    # fig.write_image("a1689_systems.png")

if __name__ == "__main__":
    # systems = du.all_a1689_systems
    # systems = du.a1689_quads
    # systems = du.a1689_7_sisters
    systems = du.a1689_3_miscreants
    main(systems)