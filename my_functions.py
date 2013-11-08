import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags


def StaggeredMesh2dSquare(n):
    """
    """

    ## Assemble mesh coordinate arrays
    dx = 1.0/n  # cell size in x,y
    xf = np.linspace(0., 1., n+1)
    # cell face coordinate vector, 1D
    xc = np.linspace(dx/2., 1.-dx/2, n)
    xb = np.hstack([0, xc, 1])  # cell center coordinate vector incl. boundaries
    Xu, Yu = np.meshgrid(xf, xb)  # u-grid coordinate arrays
    Xv, Yv = np.meshgrid(xb, xf)  # v-grid coordinate arrays
    Xp, Yp = np.meshgrid(xc, xc)  # p-grid coordinate arrays
    Xi, Yi = np.meshgrid(xf, xf)  # fd-grid coordinate arrays

    ## Plot mesh arrangement
    if (n < 12):  # plot mesh if fewer than 12 cells in x,y
        fig1 = plt.figure(117)
        fig1.clf()
        ax = fig1.add_subplot(1, 1, 1)
        ax.plot(Xi, Yi, 'k--', Yi, Xi, 'k--', Xu, Yu, 'b>',
                Xv, Yv, 'r^', Xp, Yp, 'go', Xi, Yi, 'y*')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Illustration of u-, v-, p- and fd-grid')
        fig1.show()

    return dx, Xu, Yu, Xv, Yv, Xp, Yp, Xi, Yi


def Hfunctions(n, dx, Re, u, v):
    """
    """

    m = n

    ## Compute H1,H2 functions
    uP = np.zeros((m, n))
    vP = np.zeros((m, n))
    uxP = np.zeros((m, n))
    vyP = np.zeros((m, n))

    # u,v,ux,vy to P-grid
    uP = 0.5 * (u[1:m+1, 1:n+1] + u[1:m+1, 0:n])
    uxP = 1/dx * (u[:, 1:n+1] - u[:, 0:n])
    vP = 0.5 * (v[1:m+1, 1:n+1] + v[0:m, 1:n+1])
    vyP = 1/dx * (v[1:m+1, :] - v[0:m, :])

    # u,v,uy,vx to FD-grid
    uFD = np.zeros((m+1, n+1))
    uyFD = np.zeros((m+1, n+1))
    vFD = np.zeros((m+1, n+1))
    vxFD = np.zeros((m+1, n+1))

    uFD[0, :] = u[0, :]
    uFD[1:m+1, :] = 0.5 * (u[2:m+2, :] + u[1:m+1, :])
    uFD[m, :] = u[m+1, :]

    vFD[:, 0] = v[:, 0]
    vFD[:, 1:n+1] = 0.5 * (v[:, 2:n+2] + v[:, 1:n+1])
    vFD[:, n] = v[:, n+1]

    uyFD[0, :] = 2./dx * (u[1, :] - u[0, :])
    uyFD[1:m, :] = 1./dx * (u[2:m+1, :] - u[1:m, :])
    uyFD[m, :] = 2./dx * (u[m+1, :] - u[m, :])

    vxFD[:, 0] = 2./dx * (v[:, 1] - v[:, 0])
    vxFD[:, 1:m] = 1./dx * (v[:, 2:n+1] - v[:, 1:n])
    vxFD[:, n] = 2./dx * (v[:, n+1] - v[:, n])

    ## Compute H1
    h1 = np.zeros((m, n+1))
    h1[:, 1:n] = (1./Re * (uxP[1:m+1, 1:n] - uxP[1:m+1, 0:n-1])  # 1/Re ux|_w^e
                  - (uP[:, 1:n]**2 - uP[:, 0:n-1]**2)) * 1./dx \
        + 1./dx * (1.0/Re * (uyFD[1:m+1, 1:n] - uyFD[0:m, 1:n])  # 1/Re uy|_s^n
                   - (vFD[1:m+1, 1:n] * uFD[1:m+1, 1:n]
                      - vFD[0:m, 1:n] * uFD[0:m, 1:n]))

    ## Compute H2
    h2 = np.zeros((m+1, n))
    h2[1:m, :] = (1./Re * (vxFD[1:m, 1:n+1] - vxFD[1:m, 0:n])
                  - (uFD[1:m, 1:n+1]*vFD[1:m, 1:n+1]
                     - uFD[1:m, 0:n]*vFD[1:m, 0:n])) * 1./dx \
        + 1./dx * (1./Re * (vyP[1:m, 1:n+1] - vyP[0:m-1, 1:n+1])
                   - (vP[1:m, :]**2 - vP[0:m-1, :]**2))

    return h1, h2


def LaplaceMatrix(n):
    """
    Assemble Laplacian operator matrix
    """

    ## Coefficient arrays
    D = np.ones((n, n))
    a_w = D.copy()  # use copy here to avoid pointer assignment
    a_e = D.copy()
    a_s = D.copy()
    a_n = D.copy()
    a_p = -(a_w + a_e + a_s + a_n)

    ## Impose boundary conditions and compute source array from BC
    # homogenous Neumann Px = 0 on east and west, Py = 0 on north and south
    # with CDS ghost point approach

    # west
    a_p[:, 0] = a_p[:, 0] + a_w[:, 0]
    a_w[:, 0] = 0

    # east
    a_p[:, n-1] = a_p[:, n-1] + a_e[:, n-1]
    a_e[:, n-1] = 0

    # south
    a_p[0, :] = a_p[0, :] + a_s[0, :]
    a_s[0, :] = 0

    # north
    a_p[n-1, :] = a_p[n-1, :] + a_n[n-1, :]
    a_n[n-1, :] = 0

    ## Assemble system matrix
    offsets = np.array([-n, -1, 0, 1, n])
    data = np.hstack([
        np.vstack([a_w.reshape(n**2, 1, order='F')[n:n**2], np.zeros((n, 1))]),
        np.vstack([a_s.reshape(n**2, 1, order='F')[1:n**2], np.zeros((1, 1))]),
        a_p.reshape(n**2, 1, order='F')[0:n**2],
        np.vstack([np.zeros((1, 1)),
                   a_n.reshape(n**2, 1, order='F')[0:n**2-1]]),
        np.vstack([np.zeros((n, 1)),
                   a_e.reshape(n**2, 1, order='F')[0:n**2-n]])
    ])
    return spdiags(data.T, offsets, n**2, n**2, format="csc")


def Source(n, dx, H1, H2):
    """
    Generate source array
    Note that NS2dHfunctions returns H1 and H2
    such that H1w = H1e = H2s = H2n = 0 (BC cv's)
    """
    return (dx * (H1[:, 1:n+1] - H1[:, 0:n] + H2[1:n+1, :] - H2[0:n, :]))


def postprocess(n, u, v, P):

    ## Coordinate arrays
    dx, Xu, Yu, Xv, Yv, Xp, Yp, Xi, Yi = StaggeredMesh2dSquare(n)

    ## Compute vorticity on FD-grid
    uyFD = 1./dx*(u[1:n+2, :] - u[0:n+1, :])
    vxFD = 1./dx*(v[:, 1:n+2] - v[:, 0:n+1])
    omega = vxFD - uyFD

    ## Extrapolate pressure field to walls
    Xpp, Ypp = np.meshgrid(Xv[0, :], Yu[:, 0])
    PP = np.zeros((n+2, n+2))
    PP[1:n+1, 1:n+1] = P.copy()

    # 2nd order Neumann:
    PP[1:n+1, 0] = P[:, 0]

    # 2nd order Neumann:
    PP[1:n+1, n+1] = P[:, n-1]

    # 2nd order Neumann:
    PP[0, 1:n+1] = P[0, :]

    # 2nd order Neumann:
    PP[n+1, 1:n+1] = P[n-1, :]

    return Xpp, Ypp, PP, omega
