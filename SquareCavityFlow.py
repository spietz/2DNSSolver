# -*- coding: utf-8 -*-
#!/usr/bin/python2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.sparse.linalg import splu
from scipy.interpolate import interp2d

import my_functions

## Input
n = 61  # number of cells along x,y-axis
Re = 100.0  # global Reynolds number
Ulid = -1.0  # lid velocity (+-1)
maxstep = int(1e6)  # maximum number of explicit Euler time steps
steadytol = 1.0e-6  # tolerance on dUmax/dt to reach steady solution
statstride = 1e2  # stride for writting status to screen

## Grid arrays
# Assemble staggered grid coordinate arrays
dx, Xu, Yu, Xv, Yv, Xp, Yp, Xi, Yi = my_functions.StaggeredMesh2dSquare(n)

# Time step - based on stability analysis
dt = np.min([(dx**2)*Re/4,
            1./(Re*Ulid**2)])

# Set initial flow field such that du/dx+dv/dy = 0
u = np.zeros((n+2, n+1))  # initial horizontal velocity component
v = np.zeros((n+1, n+2))  # initial vertical velocity component
u[n+1, :] = Ulid  # set lid velocity in u-array

# Various other arrays
P = np.zeros((n, n))  # alloc pressure array
gmchist = np.zeros(maxstep)  # alloc global mass conservation hist. vector
cmchist = np.zeros(maxstep)  # alloc max. cell mass conserva. hist. vector
steadyhist = np.zeros(maxstep)  # alloc global steady solution hist. vector

## Assemble and factorize Laplacian operator matrix
A = my_functions.LaplaceMatrix(n)

# LU factorization
luA = splu(A)  # Object, which has a solve method.

## Time integrate NS-equations to reach steady solution
step = 0
change = 1
while step < maxstep and change > steadytol:

    # Increment step count
    step = step + 1

    # Compute H1, H2 from momentum equations
    H1, H2 = my_functions.Hfunctions(n, dx, Re, u, v)

    # Compute rhs of Poisson eqn on p-grid
    S = my_functions.Source(n, dx, H1, H2)
    
    # Solve system
    P = luA.solve(S.reshape(n**2, order='F'))
    # P = spsolve(A, S.reshape(n**2, 1, order='F'))
    P = (P - np.mean(P)).reshape(n, n, order='F')

    # Evaluate change in velocity field gradients
    dudt = H1[:, 1:n] - 1./dx * (P[:, 1:n] - P[:, 0:n-1])
    dvdt = H2[1:n, :] - 1./dx * (P[1:n, :] - P[0:n-1, :])
    change = max(np.max(dudt), np.max(dvdt))
    steadyhist[step-1] = change

    # Step forward interiour cells
    u[1:n+1, 1:n] = u[1:n+1, 1:n] + dt*dudt
    v[1:n, 1:n+1] = v[1:n, 1:n+1] + dt*dvdt

    # Local mass conservation -> each cv
    LMC = 1./dx * (u[1:n+1, 1:n+1] - u[1:n+1, 0:n]
                   + v[1:n+1, 1:n+1] - v[0:n, 1:n+1])
    cmchist[step-1] = np.max(np.max(LMC, 0), 0)
    gmchist[step-1] = np.sum(np.sum(LMC, 0))
    
    if (step % statstride) == 0:
        print('Step=%i, change=%3.2e' % (step, change))


## Interpolate functions to P-grid
# note that interp2d works for cartesian grids only and take in 1d grids
f_int_u = interp2d(Xu[0, :], Yu[:, 0].T, u, kind='linear')
f_int_v = interp2d(Xv[0, :], Yv[:, 0].T, v, kind='linear')

# Extend pressure + vorticity
Xpp, Ypp, PP, omega = my_functions.postprocess(n, u, v, P)


"""Streamfunctions
Defined as:
(u,v) = (dψ/dy, dpsi/dx)

Thus
ψ(x,y) = ∫_y0^y u(x,ξ) dξ
or
ψ(x,y) = -∫_x0^x v(ξ,y) dξ

We don't need some fancy numerical integration scheme. Summing is enough because
we explicit state that u is constant over a cell face. Since mass is conserved,
the cumulative sum should equal zero at the upper boundary.
"""

# Manual cumsum
psi = np.zeros((n+1, n+1))
for i in range(n):
    for j in range(n):
        psi[i+1, j+1] = psi[i, j+1] + u[i+1, j+1]*dx

# same just shorter
psi2 = np.cumsum(u*dx,0)
psi2 = psi2[:-1]

psi3 = -np.cumsum(v*dx,1)
psi3 = psi3[:,:-1]


## Plot results
plt.ion()  # turn on interactive mode

fig1 = plt.figure(1)
fig1.clf()

# Streamlines
# Contours of the stream function:
# Set contour levels. From the article "High-Re Solutions for Incompressible
# Using the Navier-Stokes Equations Multigrid Method".
# By U. GHIA, K. N. GHIA, AND C. T. SHIN
if int(Re) == 1000:
    levels = np.array([
        -1e-10, -1e-07, -1e-05, -1e-04, -1e-03, -0.0100, -0.0300, -0.0500,
        -0.0700, -0.0900, -0.1000, -0.1100, -0.1150, -0.1175, 1.0e-8, 1.0e-7,
        1.0e-6, 1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4, 1.0e-3, 1.5e-3, 3.0e-3])
    # because Ulid = -1, whereas the article have Ulid = 1, we need the levels
    # with opposite signs. Remember that sign change for the streamfunction
    # means the flow is flowing in the opposite direction.
    levels = - levels
else:
    levels = np.array([-0.1, -0.08, -0.06, -0.04, -0.02, -0.01, -3e-3,
                       -1e-3, -3e-4, -1e-4, -3e-5, -1e-5, -3e-6, -1e-6, -1e-7, -1e-8,
                       -1e-9, -1e-10, 0.0 , 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5,
                       3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01 , 0.03 , 0.05 , 0.07 , 0.09,
                       0.1 , 0.11 , 0.115, 0.1175])

ax1 = fig1.add_subplot(2, 2, 1)
ax1.contour(Xi, Yi, psi, levels, colors='k', linewidths=0.5)
ax1.set_title('Stream function')
ax1.set_xlabel('x/L')
ax1.set_ylabel('y/L')
ax1.set_aspect('equal')
plt.tight_layout()
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
plt.tight_layout()

# Pressure
ax2 = fig1.add_subplot(2, 2, 2)
p2 = ax2.contourf(Xpp, Ypp, PP, cmap=cm.bwr)
ax2.set_title('Dynamic pressure')
ax2.set_xlabel('x/L')
ax2.set_ylabel('y/L')
fig1.colorbar(p2, ax=ax2)
ax2.set_aspect('equal')
plt.tight_layout()

# Vorticity
ax3 = fig1.add_subplot(2, 2, 3)
p3 = ax3.contourf(Xi, Yi, omega, cmap=cm.bwr)
ax3.set_title('Vorticity')
ax3.set_xlabel('x/L')
ax3.set_ylabel('y/L')
fig1.colorbar(p3, ax=ax3)
ax3.set_aspect('equal')
plt.tight_layout()

# Steady state convergence
ax4 = fig1.add_subplot(2, 2, 4)
ax4.plot(range(step), steadyhist[:step])
ax4.set_title('Change history')
ax4.set_xlabel('step')
ax4.set_ylabel(r'$\max{|\frac{du_i}{dt}|}$')
ax4.grid(True)
ax4.set_yscale('log')
plt.tight_layout()
