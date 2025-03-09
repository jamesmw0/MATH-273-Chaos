import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint

# Define the Lorenz system
def lorenz(state, t, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dxdt = sigma*(y - x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return [dxdt, dydt, dzdt]

# Time array and initial condition
t = np.linspace(0, 40, 10000)
init_state = [1, 1, 1]
sol = odeint(lorenz, init_state, t)

# Create interactive 3D plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=sol[:,0],
    y=sol[:,1],
    z=sol[:,2],
    mode='lines',
    line=dict(width=2, color=np.linspace(0, 1, len(sol)))
)])
fig.update_layout(
    title="Lorenz Attractor",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)
fig.show()

# Assuming you have a Plotly figure object called "fig"
fig.write_html("my_interactive_plot.html")



# Create data for a sphere (level set V(x,y,z)=0.5, or equivalently x^2+y^2+z^2=1)
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

# Create interactive 3D surface plot using Plotly
fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='Viridis')])
fig.update_layout(
    title='Liapunov Function Level Set: x² + y² + z² = 1',
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z"
    )
)
fig.show()