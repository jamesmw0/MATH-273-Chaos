import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] = "C:/Program Files/ffmpeg-7.1-essentials_build/ffmpeg-7.1-essentials_build/bin/ffmpeg.exe"

# Define Lorenz system equations
def lorenz(state, t, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Initial conditions and time array for integration
init_state = [1, 1, 1]
t = np.linspace(0, 40, 10000)

# Constant parameter values for beta and rho
DEFAULT_BETA = 8/3
DEFAULT_RHO = 28

# Set up figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=30, azim=45)

# Compute initial solution w/ initial sigma value
initial_sigma = 10
sol = odeint(lorenz, init_state, t, args=(initial_sigma, DEFAULT_BETA, DEFAULT_RHO))
line, = ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], lw=0.5)
ax.set_title(f"Lorenz Attractor: σ={initial_sigma:.2f}, β={DEFAULT_BETA:.2f}, ρ={DEFAULT_RHO:.2f}")

# Add a red ball to mark the end point of the trajectory
ball = ax.scatter(sol[-1, 0], sol[-1, 1], sol[-1, 2], color='red', s=20)

# Animation update function: vary sigma from 0.1 to 20 in 0.02
def update(frame):
    sigma = 0.1 + (frame % 996) * 0.02
    sol = odeint(lorenz, init_state, t, args=(sigma, DEFAULT_BETA, DEFAULT_RHO))
    line.set_data(sol[:, 0], sol[:, 1])
    line.set_3d_properties(sol[:, 2])
    # Update the red ball to the last point in the trajectory
    ball._offsets3d = (sol[-1:, 0], sol[-1:, 1], sol[-1:, 2])
    current_azim = 45 + (frame * 0.1) % 360
    ax.view_init(elev=30, azim=current_azim)
    ax.set_title(f"Lorenz Attractor: σ={sigma:.2f}, β={DEFAULT_BETA:.2f}, ρ={DEFAULT_RHO:.2f}")
    return line, ball

# Create the animation: update every 15 ms for a fast update rate
ani = animation.FuncAnimation(fig, update, frames=1000, interval=15, blit=False)

# Save animation as an MP4 file using ffmpeg
ani.save("C:/Users/james/Downloads/MATH 273/lorenz_simulation_v2.mp4", writer="ffmpeg", fps=50)
global saved_animation
saved_animation = ani

plt.show()