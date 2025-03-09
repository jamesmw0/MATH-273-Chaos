import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def hyperbolic_system(state, t):
    x, y = state
    dxdt = 2*x
    dydt = -y
    return [dxdt, dydt]

# Create a grid for the vector field
x_vals = np.linspace(-2, 2, 20)
y_vals = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x_vals, y_vals)
U = 2*X
V = -Y

plt.figure(figsize=(6,6))
plt.quiver(X, Y, U, V, color='blue')
plt.title("Phase Portrait of a Hyperbolic Point")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(0, 0, 'ro', label="Equilibrium (0,0)")

# Simulate trajectories starting from different initial conditions
t = np.linspace(0, 2, 200)
initial_conditions = [[0.5, 0.5], [-0.5, 0.5], [1, -0.5], [-1, -1]]
for ic in initial_conditions:
    sol = odeint(hyperbolic_system, ic, t)
    plt.plot(sol[:,0], sol[:,1], 'r')

plt.legend()
plt.show()


r_vals = np.linspace(-1, 2, 400)
# Equilibrium at x = 0 is always present.
x_zero = np.zeros_like(r_vals)
# For r>=0, additional equilibria are x = ±sqrt(r)
x_plus = np.array([np.sqrt(r) if r >= 0 else np.nan for r in r_vals])
x_minus = np.array([-np.sqrt(r) if r >= 0 else np.nan for r in r_vals])

plt.figure(figsize=(6,4))
plt.plot(r_vals, x_zero, 'k', label="x = 0")
plt.plot(r_vals, x_plus, 'r', label="x = +sqrt(r)")
plt.plot(r_vals, x_minus, 'r', label="x = -sqrt(r)")
plt.title("Bifurcation Diagram for dx/dt = r*x - x^3")
plt.xlabel("Parameter r")
plt.ylabel("Equilibrium x")
plt.legend()
plt.show()


def system(state, t):
    x, y = state
    dxdt = x*(1-y)
    dydt = y*(x-1)
    return [dxdt, dydt]

# Create a grid for the vector field
x_vals = np.linspace(-2, 4, 20)
y_vals = np.linspace(-2, 4, 20)
X, Y = np.meshgrid(x_vals, y_vals)
U = X*(1-Y)
V = Y*(X-1)

plt.figure(figsize=(6,6))
plt.quiver(X, Y, U, V, color='blue')
plt.title("Phase Portrait with Nullclines")
plt.xlabel("x")
plt.ylabel("y")

# Plot the nullclines
plt.axvline(x=0, color='red', linestyle='--', label='x = 0')
plt.axhline(y=0, color='green', linestyle='--', label='y = 0')
plt.axhline(y=1, color='purple', linestyle='--', label='y = 1')
plt.axvline(x=1, color='orange', linestyle='--', label='x = 1')

# Simulate some trajectories
t = np.linspace(0, 10, 400)
initial_conditions = [[0.5, 0.5], [2, 0.5], [3, 2]]
for ic in initial_conditions:
    sol = odeint(system, ic, t)
    plt.plot(sol[:,0], sol[:,1], 'r')

plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def limit_cycle(state, t):
    x, y = state
    r2 = x**2 + y**2
    dxdt = (1 - r2)*x - y
    dydt = (1 - r2)*y + x
    return [dxdt, dydt]

t = np.linspace(0, 30, 600)
plt.figure(figsize=(6,6))
# Plot the expected limit cycle (unit circle)
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta), 'k--', label="Limit Cycle (r = 1)")

# Simulate trajectories from various initial conditions
initial_conditions = [[0.1, 0.1], [0.5, 0], [1.5, 0], [0, -1.5]]
for ic in initial_conditions:
    sol = odeint(limit_cycle, ic, t)
    plt.plot(sol[:,0], sol[:,1], label=f"Initial: {ic}")

plt.title("Limit Cycle and ω-limit set")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis('equal')
plt.show()


def stable_system(state, t):
    x, y = state
    dxdt = -x
    dydt = -y
    return [dxdt, dydt]

# Define the Liapunov function V(x,y) = 0.5*(x^2+y^2)
def V(x, y):
    return 0.5*(x**2 + y**2)

x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = V(X, Y)

plt.figure(figsize=(6,6))
contours = plt.contour(X, Y, Z, levels=20)
plt.clabel(contours, inline=True, fontsize=8)
plt.title("Contours of Liapunov Function V(x,y) = 0.5*(x² + y²)")
plt.xlabel("x")
plt.ylabel("y")

# Simulate trajectories from different starting points
t = np.linspace(0, 5, 200)
initial_conditions = [[1.5, 1.5], [1, -1], [-1, 1], [-1.5, -1]]
for ic in initial_conditions:
    sol = odeint(stable_system, ic, t)
    plt.plot(sol[:,0], sol[:,1], label=f"Initial: {ic}")

plt.legend()
plt.axis('equal')
plt.show()

