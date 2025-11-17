import pandas as pd
import numpy as np
import plotly.express as px
import util.data as du
import util.math as mu

from PIL import Image
import os
import gifmaker

from tqdm import tqdm

quad_ids_map = ['4', '8', '9', '19']

# 'quad_idx', 'image_id', 'x', 'y', 'ra', 'dec', 'asym_angle', 'axis_ratio', 'error', 'center_x', 'center_y', 'noise'
metric = 'axis_ratio'  # 'error', 'asym_angle', 'axis_ratio', 'center_x', 'center_y'

def main(metric='error'):
    df = pd.read_pickle('results_500_ffits_lim.pkl')
    batches = df.groupby(['quad_idx', 'image_id'])
    
    for (quad_idx, image_id), batch in batches:
        quad_idx = int(quad_idx)
        image_id = int(image_id)
        batch = batch.reset_index(drop=True)
        row_with_noise_zero = batch[batch['noise'] == 0]
        
        fig = px.line(
            batch, x='noise', y=metric,
            title=f'Quad {quad_ids_map[quad_idx]} Image {image_id}',
            markers=True
        )
        fig.update_traces(
            mode='markers+lines',
            marker=dict(size=5),
            line=dict(width=2)
        )
        fig.update_layout(
            xaxis_title='Noise',
            yaxis_title=metric.replace('_', ' ').title(),
            width=800,
            height=600
        )
        
        fig.write_image(f'plots/a1689/statistical_modeling/individual/500_noise_vs_{metric}_{quad_ids_map[quad_idx]}_{image_id}.png')
    
def alt(metric):
    df = pd.read_pickle('results_fits_lim.pkl')

    # one label per (quad_idx, image_id) pair
    df['quad_image'] = df['quad_idx'].astype(str) + '_' + df['image_id'].astype(str)

    # make sure points inside each pair are drawn in x-order
    df = df.sort_values(['quad_idx', 'image_id', 'noise'])

    fig = px.line(
        df,
        x='noise',
        y=metric,
        color='quad_image',      # <-- unique colour per pair
        line_group='quad_image', # <-- keep lines separate
        markers=True,
        title=f"Noise vs {metric.replace('_',' ').title()} per Quad–Image Pair",
    )

    fig.update_traces(marker=dict(size=5), line=dict(width=2))
    fig.update_layout(
        legend_title='quad_idx-image_id',
        xaxis_title='Noise',
        yaxis_title=metric.replace('_', ' ').title(),
        width=800,
        height=600,
    )

    fig.write_image(f'plots/a1689/statistical_modeling/500_noise_vs_{metric}.png')
    
    
def alt2():
    df = pd.read_pickle('ellipse_results_500_fits_lim.pkl')

    # one label per (quad_idx, image_id) pair
    df['quad_image'] = df['quad_idx'].astype(str) + '_' + df['image_id'].astype(str)
    
    graph_limits = np.array([[-2000, -1300], [2000, 1600]])
    gx_min, gx_max = graph_limits[:, 0]

    # make sure points inside each pair are drawn in x-order
    batches = df.groupby('quad_image')
    for quad_image, batch in batches:
        quad_id = quad_ids_map[int(quad_image.split('_')[0][0])]
        image_id = int(quad_image.split('_')[1][0])
        print(f'Processing quad {quad_id}, image {image_id}…')
        
        batch = batch.sort_values('displacement')
        
        first_x = batch['x'].iloc[0]
        first_y = batch['y'].iloc[0]
        
        last_x = batch['x'].iloc[-1]
        last_y = batch['y'].iloc[-1]
        
        linspace_x = np.linspace(first_x, last_x, 500)
        linspace_y = np.linspace(first_y, last_y, 500)
        
        for i, row in tqdm(batch.iterrows()):
            displacement = row['displacement']
            ell_coeffs = row['ell_coeffs']
            x = row['x']
            y = row['y']
            
            xs1, ys1, xs2, ys2 = mu.generate_conic_linspaces(
                ell_coeffs, 5000, gx_min, gx_max
            )
            
            fig = px.scatter(
                x=linspace_x, y=linspace_y,
                title=f"Graph for quad: {quad_id}, image: {image_id}, displacement: {displacement} (axis ratio: {row['axis_ratio']:.5f})",
                labels={'x': 'X', 'y': 'Y'}
            )
            fig.update_traces(mode='lines', line=dict(width=1))  # Set thin line width
            fig.add_scatter(x=xs1, y=ys1, mode='lines', name='Wynne Ellipse', line=dict(color='blue'))
            fig.add_scatter(x=xs2, y=ys2, mode='lines', name='Wynne Ellipse', line=dict(color='blue'))
            fig.add_scatter(x=[x], y=[y], mode='markers', name=f'Displacement {displacement}')
            
            fig.update_layout(showlegend=False)
            
            fig.update_layout(
                xaxis_title='X',
                yaxis_title='Y',
                width=800,
                height=600,
                xaxis=dict(range=[gx_min, gx_max]),
                yaxis=dict(range=graph_limits[:, 1]),
                yaxis_scaleanchor="x"  # Ensures equal scaling between x and y axes
            )
            
            os.makedirs(f'plots/a1689/statistical_modeling/individual 500/{quad_id}_{image_id}', exist_ok=True)
            fig.write_image(f'plots/a1689/statistical_modeling/individual 500/{quad_id}_{image_id}/{i}_{displacement}.png')
            
        # Load all images from the folder and save them as a GIF
        image_folder = f'plots/a1689/statistical_modeling/individual 500/{quad_id}_{image_id}'
        images = []
        for file_name in sorted(os.listdir(image_folder)):
            if file_name.endswith('.png'):
                file_path = os.path.join(image_folder, file_name)
                images.append(Image.open(file_path))
        
        gif_path = f'plots/a1689/statistical_modeling/individual 500/{quad_id}_{image_id}_animation.gif'
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=1,  # Duration for each frame in milliseconds
            loop=0         # Loop count (0 means loop indefinitely)
        )
            
if __name__ == '__main__':
    # main(metric)
    # alt(metric)
    alt2()