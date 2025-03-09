import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import odeint

app = dash.Dash(__name__)
server = app.server 

# Define default parameter values
DEFAULT_SIGMA = 10
DEFAULT_BETA = 8/3
DEFAULT_RHO = 28

app.layout = html.Div([
    html.H1("Interactive Lorenz Attractor"),
    
    html.Div([
        html.Label("Sigma (σ):"),
        dcc.Slider(
            id='sigma-slider',
            min=0.1,
            max=20,
            step=0.1,
            value=DEFAULT_SIGMA,
            marks={i: str(i) for i in range(1, 21)}
        )
    ], style={'padding': '10px'}),
    
    html.Div([
        html.Label("Beta (β):"),
        dcc.Slider(
            id='beta-slider',
            min=0.1,
            max=10,
            step=0.1,
            value=DEFAULT_BETA,
            marks={i: str(i) for i in range(1, 11)}
        )
    ], style={'padding': '10px'}),
    
    html.Div([
        html.Label("Rho (ρ):"),
        dcc.Slider(
            id='rho-slider',
            min=0.1,
            max=50,
            step=0.1,
            value=DEFAULT_RHO,
            marks={i: str(i) for i in range(5, 51, 5)}
        )
    ], style={'padding': '10px'}),
    
    # "Default" button to reset parameters
    html.Button('Default', id='default-button', n_clicks=0, style={'margin': '10px'}),
    
    # Fixing camera settings
    dcc.Store(id='camera-store', data={'camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}}}),

    # Increase graph size by setting height and width
    dcc.Graph(id='lorenz-graph', style={'height': '65vh', 'width': '100%'})
])

# Callback to update the graph based on slider values and use stored camera view
@app.callback(
    Output('lorenz-graph', 'figure'),
    Input('sigma-slider', 'value'),
    Input('beta-slider', 'value'),
    Input('rho-slider', 'value'),
    State('camera-store', 'data')
)
def update_graph(sigma, beta, rho, camera_data):
    # Lorenz system equations
    def lorenz(state, t):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
    
    # Time array and integration
    t = np.linspace(0, 40, 10000)
    init_state = [1, 1, 1]
    sol = odeint(lorenz, init_state, t)
    
    # Create a 3D scatter plot for the Lorenz attractor
    fig = go.Figure(data=[go.Scatter3d(
        x=sol[:, 0],
        y=sol[:, 1],
        z=sol[:, 2],
        mode='lines',
        line=dict(width=2, color=sol[:, 2], colorscale='Viridis')
    )])
    
    # Use the stored camera settings, or default if not available
    camera = camera_data.get('camera', {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}})
    
    fig.update_layout(
        title=f"Lorenz Attractor: σ={sigma}, β={beta}, ρ={rho}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=camera
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

# Callback to reset sliders to default values when the "Default" button is clicked
@app.callback(
    [Output('sigma-slider', 'value'),
     Output('beta-slider', 'value'),
     Output('rho-slider', 'value')],
    Input('default-button', 'n_clicks')
)
def reset_defaults(n_clicks):
    return DEFAULT_SIGMA, DEFAULT_BETA, DEFAULT_RHO

# Callback to update camera-store when the graph's layout changes (user rotates the view)
@app.callback(
    Output('camera-store', 'data'),
    Input('lorenz-graph', 'relayoutData'),
    State('camera-store', 'data')
)
def update_camera(relayoutData, stored_data):
    if relayoutData is not None and 'scene.camera' in relayoutData:
        stored_data['camera'] = relayoutData['scene.camera']
    return stored_data

if __name__ == '__main__':
    app.run_server(debug=True)

