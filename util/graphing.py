import plotly.graph_objects as go

plotly_colors_50 = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#AA0DFE', '#3283FE', '#85660D', '#782AB6', '#565656',
    '#1C8356', '#16FF32', '#F7E1A0', '#E2E2E2', '#1E6E00',
    '#BA0DFE', '#46F0F0', '#8B0000', '#A4A65D', '#2BCE48',
    '#C96424', '#FF2F80', '#61615A', '#BAA0FE', '#6E750E',
    '#00A9F8', '#FF913F', '#938A81', '#CDC618', '#E798A3',
    '#C7144C', '#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3'
]

class GraphingFormat:
    def __init__(self, mode, format_dict, name):
        self.mode = mode
        self.format_dict = format_dict
        self.name = name

formats = {
    'hyperbola': GraphingFormat(
        mode='lines',
        format_dict=dict(color='dodgerblue'),
        name='Hyperbola'
    ),
    'ellipse': GraphingFormat(
        mode='lines',
        format_dict=dict(color='seagreen'),
        name='Ellipse'
    ),
    'asymptote': GraphingFormat(
        mode='lines',
        format_dict=dict(color='darkorange', dash='dash'),
        name='Asymptote'
    ),
    'images': GraphingFormat(
        mode='markers',
        format_dict=dict(size=13, color='crimson', symbol='circle'),
        name='Images'
    ),
    'lens': GraphingFormat(
        mode='markers',
        format_dict=dict(size=11, color='goldenrod', symbol='star'),
        name='Lens'
    ),
    'source': GraphingFormat(
        mode='markers',
        format_dict=dict(size=14, color='blueviolet', symbol='square'),
        name='Source'
    ),
    'missing_image': GraphingFormat(
        mode="markers",
        format_dict=dict(size=11, color="mediumorchid", symbol="diamond"),
        name='4th Image'
    ),
    'intersection': GraphingFormat(
        mode='markers',
        format_dict=dict(size=6, color='black', symbol='x'),
        name='Intersection'
    ),
    'predicted_center': GraphingFormat(
        mode='markers',
        format_dict=dict(size=11, color='royalblue', symbol='cross'),
        name='Predicted Center'
    )
}

def get_graphing_format(format_type, color_idx=None, line_width=None):
    if format_type not in formats:
        raise ValueError(f"Unknown format type: {format_type}")
    formatting = formats[format_type]
    if color_idx is not None:
        formatting.format_dict['color'] = plotly_colors_50[color_idx % len(plotly_colors_50)]
    if line_width is not None and formatting.mode == 'lines':
        formatting.format_dict['width'] = line_width
    return formatting

class Graph:
    def __init__(self, title, xaxis_title='x', yaxis_title='y'):
        self.fig = go.Figure()
        self.fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            showlegend=True,
            template='plotly_white'
        )

    def show(self):
        self.fig.show()

    def save(self, filename):
        self.fig.write_image(filename)

    def add_trace(self, x, y, format_type, color_idx=None, line_width=None, showlegend=True, name=None):
        # Ensure x and y are iterable, add dimension if scalar
        if not hasattr(x, '__iter__'):
            x = [x]
        if not hasattr(y, '__iter__'):
            y = [y]
        
        formatting = get_graphing_format(format_type, color_idx, line_width)
            
        self.fig.add_trace(go.Scatter(
            x=x, y=y,
            mode=formatting.mode,
            line=formatting.format_dict if formatting.mode == 'lines' else None,
            marker=formatting.format_dict if formatting.mode == 'markers' else None,
            name=name if name else formatting.name,
            showlegend=showlegend
        ))
    
    def update_axes(self, xaxis_range=None, yaxis_range=None):
        if xaxis_range is not None:
            self.fig.update_xaxes(range=xaxis_range, scaleanchor='y', scaleratio=1)
        if yaxis_range is not None:
            self.fig.update_yaxes(range=yaxis_range)
        
    def enable_legend(self, legend=None):
        self.fig.update_layout(
            showlegend = True,
            legend = legend if legend is not None else dict(
                orientation='h',
                yanchor='bottom', y=1.02,
                xanchor='right', x=1
            )
        )
        
    def annotate_quads(self, xy_pairs, name):
        for xs, ys in xy_pairs:
            if xs.size == 0 or ys.size == 0:
                continue

            # Check first and last point
            for x, y in ((xs[0], ys[0]), (xs[-1], ys[-1])):
                if x in (2000, -2000):
                    self.fig.add_annotation(
                        x=x, y=y,
                        text=f'Quad {name}',
                        showarrow=True,
                        arrowhead=2,
                        ax=30 if x == 2000 else -30,
                        ay=0
                    )