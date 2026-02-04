#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
#  TP - Introduction à l'interpolation spatiale et aux géostatistiques  #
#########################################################################

# P. Bosser / ENSTA Bretagne
# Version du 26/02/2024


# Numpy
import numpy as np
# Matplotlib / plot
import matplotlib.pyplot as plt
    
from matplotlib import cm

################## Modèle de fonction d'interpolation ##################

def interp_xxx(x_obs, y_obs, z_obs, x_int, y_int):
    # Interpolation par ???
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    
    z_int = np.nan*np.zeros(x_int.shape)
    #
    # ...
    #
    return z_int




####################### Fonctions d'interpolation ######################


##LIN##
def interp_lin(x_obs, y_obs, z_obs, x_int, y_int):
    # Interpolation par ???
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    
    from scipy.spatial import Delaunay as delaunay
    
    z_int = np.nan*np.zeros(x_int.shape)
    
    # On construit la triangulation ; tri est un tableau de 3 colonnes, le nombre de ligne correspond au nombres de 
    # triangles
    points = np.column_stack((x_obs[:,0], y_obs[:,0]))
    tri = delaunay(points)


    for i in range(x_int.shape[0]):
<<<<<<< HEAD
        for j in range(y_int.shape[0]):
            # on recherche le numéro du triangle dans 
            idx_t = tri.find_simplex( np.array([x_int[i], y_int[j]]) )
=======
        
        # on recherche le numéro du triangle dans 
        pt = np.array([[x_int[i], y_int[i]]])
        idx_t = tri.find_simplex(pt)[0]
>>>>>>> 3f6fe6bd703d88d5480ec6e79937ba8e294bb546

            if idx_t != -1:
                

                # on récupère les numéros des sommets du triangle contenant le point (x0,y0)
                idx_s = tri.simplices[idx_t,:]

                # x_obs, y_obs sont des tableaux à 2 dimensions ; il faut les préciser pour en extraire un scalaire
                x1 = x_obs[ idx_s[0],0 ] ; y1 = y_obs[ idx_s[0],0 ]
                x2 = x_obs[ idx_s[1],0 ] ; y2 = y_obs[ idx_s[1],0 ]
                x3 = x_obs[ idx_s[2],0 ] ; y3 = y_obs[ idx_s[2],0 ]

                z1 = z_obs[ idx_s[0],0 ]
                z2 = z_obs[ idx_s[1],0 ]
                z3 = z_obs[ idx_s[2],0 ]


                mat = np.array([[x1,y1,1],[x2,y2,1],[x3,y3,1]])
                z_mat = np.array([z1,z2,z3])

                a, b, c = np.linalg.solve(mat, z_mat)

                z_int[i] = a*x_int[i] + b*y_int[j] + c


    return z_int

    
##PPV##

def interp_ppv(x_obs, y_obs, z_obs, x_int, y_int):
    # Interpolation par plus proche voisin
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    print(np.__version__)
    z_int = np.nan*np.zeros(x_int.shape)
    for i in np.arange(0,x_int.shape[0]):
        d = np.sqrt((x_int[i]-x_obs)**2+(y_int[i]-y_obs)**2)
        idx = np.argmin(d)
        z_int[i] = z_obs[idx,0]
    return z_int


##INVERSE DES DISTANCES##

def interp_inv(x_obs, y_obs, z_obs, x_int, y_int, p=2):
    # Interpolation par inverse des distances
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    # p : puissance de l'inverse des distances
    z_int = np.nan*np.zeros(x_int.shape)
    for i in np.arange(0,x_int.shape[0]):
        d = np.sqrt((x_int[i]-x_obs)**2+(y_int[i]-y_obs)**2)
        w = 1/(d**p)
        z_int[i] = np.sum(w*z_obs)/np.sum(w)
    return z_int
    
##SPLINES##

def interp_splines(x_obs, y_obs, z_obs, x_int, y_int, degree=3):
    # Interpolation par splines
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    # degree : degré des splines (1 ou 3)
    from scipy.interpolate import Rbf
    rbf = Rbf(x_obs.flatten(), y_obs.flatten(), z_obs.flatten(), function='linear' if degree==1 else 'cubic')
    z_int = rbf(x_int, y_int)
    return z_int

##MOINDRES CARRES##

def calcul_nuee(x_obs, y_obs, z_obs):
    ecarts_z = []
    distance_couple = []

    n = x_obs.shape[0]

    for i in range(n - 1):
        for j in range(i + 1, n):
            dist = np.sqrt((x_obs[i] - x_obs[j]) ** 2 +(y_obs[i] - y_obs[j]) ** 2)
            distance_couple.append(dist)

            ecart = 0.5 * (z_obs[i] - z_obs[j]) ** 2
            ecarts_z.append(ecart)

    return np.array(distance_couple), np.array(ecarts_z)



def calc_var_exp(h_raw, g_raw, hmax, nbin):
    
    bins = np.linspace(0, hmax, nbin + 1)
    h_exp = []
    g_exp = []

    for i in range(nbin):
        idx = np.where((h_raw >= bins[i]) & (h_raw < bins[i + 1]))[0]

        if len(idx) > 0:
            h_exp.append(np.mean(h_raw[idx]))
            g_exp.append(np.mean(g_raw[idx]))

    return np.array(h_exp), np.array(g_exp)

def gamma(h, c, a):
    h = np.asarray(h)
    return np.where(
        h <= a,
        c*(7*(h**2/a**2)-(35/4)*(h**3/a**3) + (7/2)*(h**5/a**5) -(3/4)*(h**7/a**7)),
        c
    )

def deriv_a(h, c, a):
    h = np.asarray(h)
    return np.where(
        h <= a,
        c*(-14*(h**2/a**3)-(35/4)*(-3)*(h**3/a**4) + (7/2)*(-5)*(h**5/a**6) -(3/4)*(-7)*(h**7/a**8)),
        0
    )
    
    
def deriv_c(h, c, a):
    h = np.asarray(h)

    return np.where(
        h <= a,
        (7*(h**2/a**2)-(35/4)*(h**3/a**3) + (7/2)*(h**5/a**5) -(3/4)*(h**7/a**7)),
        1
        )

def moindres_carres(x_obs, y_obs, z_obs, hmax, nbin) :
    h, g = calcul_nuee(x_obs, y_obs, z_obs)
    h_exp, g_exp = calc_var_exp(h, g, hmax, nbin)
    
    
    # valeur approchée a0 et C0
    c0 = max(g_exp)
    idx = np.argmax(g_exp)
    a0 = h[idx]
    
    # construction Y
    gamma_c0_a0 = gamma(h_exp, c0, a0)
    
    Y = g_exp - gamma_c0_a0
        
    d_a = deriv_a(h_exp, c0, a0)
    d_c = deriv_c(h_exp, c0, a0)
    
    A = np.ones((d_a.shape[0], 2))
    
    for i in range(d_a.shape[0]):
        A[i,0] = d_a[i]
        A[i,1] = d_c[i]
    
    inv = np.linalg.inv(A.T @ A)
    
    X_chap = inv @ A.T @ Y
    
    
    a0 += X_chap[0]
    c0 += X_chap[1]
    
    
    while abs(X_chap[0]) > 10e-3 or abs(X_chap[1]) > 10e-3  :
        
        gamma_c0_a0 = gamma(h_exp, c0, a0)
        Y = g_exp - gamma_c0_a0
        
        d_a = deriv_a(h_exp, c0, a0)
        d_c = deriv_c(h_exp, c0, a0)
        
        inv = np.linalg.inv(A.T @ A)
    
        X_chap = inv @ A.T @ Y
        
        
        a0 = a0 + X_chap[0]
        c0 = c0 + X_chap[1]

    g_fit = gamma(h_exp, c0, a0)
        
    plt.figure()
    plt.scatter(h, g, s=5, alpha=0.4, label="nuée")
    plt.plot(h_exp, g_exp, '-or', lw=2, label="variogramme exp")
    plt.plot(h_exp, g_fit, label='fit cubique', color='g')
    plt.xlabel("h")
    plt.ylabel("γ(h)")
    plt.legend()
    plt.grid()
    plt.show()

    return a0, c0

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def interp_krg(x_obs, y_obs, z_obs, x_int, y_int, c0, a0):
    z_int = np.nan * np.zeros(x_int.shape)
    z_inc = np.nan * np.zeros(x_int.shape)

    n = x_obs.shape[0]

    # résolution de AX = B pour chaque point de la grille

    # construction de A, constante pour chaque point à interpoler
    gamma_a = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = distance(x_obs[i,0], y_obs[i,0], x_obs[j,0], y_obs[j,0])
                gamma_a[i,j] = gamma(dist, c0, a0)

    gamma_A = np.zeros((n+1, n+1))
    gamma_A[:n,:n] = gamma_a
    gamma_A[:n,n] = 1
    gamma_A[n, :n] = 1
    gamma_A[n, n] = 0

    # construction de B, différent pour chaque point à interpoler

    for a in range(x_int.shape[0]):
        for b in range(y_int.shape[0]):

            dx = x_obs - x_int[a, b]
            dy = y_obs - y_int[a, b]
            h = np.sqrt(dx**2 + dy**2)
            gamma_B = np.zeros(n + 1)
            gamma_B[:-1] = gamma(h, c0, a0).squeeze()
            gamma_B[-1] = 1

            gamma_B = gamma_B.reshape((-1,1))
            lamb = np.linalg.solve(gamma_A, gamma_B)
            lamb = lamb.reshape((n+1,1))
            z_int[a,b] = np.sum(lamb[: -1]*z_obs)
            z_inc[a,b] = np. sum(lamb[: -1]*gamma_B[: -1]) + lamb[-1,0]
    
    return z_int, z_inc

############################# Visualisation ############################

def plot_contour_2d(x_grd ,y_grd ,z_grd, x_obs = np.array([]) ,y_obs = np.array([]), xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé du champ interpolé sous forme d'isolignes
    # x_grd, y_grd, z_grd : grille de valeurs interpolées
    # x_obs, y_obs : observations (facultatif)
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    
    z_grd_m = np.ma.masked_invalid(z_grd)
    fig = plt.figure()
    plt.contour(x_grd, y_grd, z_grd_m, int(np.round((np.max(z_grd_m)-np.min(z_grd_m))/4)),colors ='k')
    if x_obs.shape[0]>0:
        plt.scatter(x_obs, y_obs, marker = 'o', c = 'k', s = 5)
        dx = max(x_obs)-min(x_obs)
        dy = max(y_obs)-min(y_obs)
        minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
        miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    else:
        dx = np.max(x_grd)-np.min(x_grd)
        dy = np.max(y_grd)-np.min(y_grd)
        minx = np.min(x_grd)-0.05*dx; maxx = np.max(x_grd)+0.05*dx
        miny = np.min(y_grd)-0.05*dy; maxy = np.max(y_grd)+0.05*dy
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_surface_2d(x_grd ,y_grd ,z_grd, x_obs = np.array([]) ,y_obs = np.array([]), minmax = [0,0], xlabel = "", ylabel = "", zlabel = "", title = "", fileo = "", cmap = cm.terrain):
    # Tracé du champ interpolé sous forme d'une surface colorée
    # x_grd, y_grd, z_grd : grille de valeurs interpolées
    # x_obs, y_obs : observations (facultatif)
    # minmax : valeurs min et max de la variable interpolée (facultatif)
    # xlabel, ylabel, zlabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    # cmap : nom de la carte de couleur
    z_grd_m = np.ma.masked_invalid(z_grd)
    fig = plt.figure()
    if minmax[0] < minmax[-1]:
        p=plt.pcolormesh(x_grd, y_grd, z_grd_m, cmap=cmap, vmin = minmax[0], vmax = minmax[-1], shading = 'auto')
    else:
        p=plt.pcolormesh(x_grd, y_grd, z_grd_m, cmap=cmap, shading = 'auto')
    if x_obs.shape[0]>0:
        plt.scatter(x_obs, y_obs, marker = 'o', c = 'k', s = 5)
        dx = max(x_obs)-min(x_obs)
        dy = max(y_obs)-min(y_obs)
        minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
        miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    else:
        dx = np.max(x_grd)-np.min(x_grd)
        dy = np.max(y_grd)-np.min(y_grd)
        minx = np.min(x_grd)-0.05*dx; maxx = np.max(x_grd)+0.05*dx
        miny = np.min(y_grd)-0.05*dy; maxy = np.max(y_grd)+0.05*dy
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    fig.colorbar(p,ax=plt.gca(),label=zlabel,fraction=0.046, pad=0.04)
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_points(x_obs, y_obs, xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé des sites d'observations
    # x_obs, y_obs : observations
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    
    fig = plt.figure()
    ax = plt.gca()
    plt.plot(x_obs, y_obs, 'ok', ms = 4)
    dx = max(x_obs)-min(x_obs)
    dy = max(y_obs)-min(y_obs)
    minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
    miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_patch(x_obs, y_obs, z_obs, fig = "", minmax = [0,0], xlabel = "", ylabel = "", zlabel = "", title = "", fileo = "", cmap = cm.terrain, marker = 'o', s= 80,ec=None,lw=0, cb=True):
    # Tracé des valeurs observées
    # x_obs, y_obs, z_obs : observations
    # fig : figure sur laquelle faire le tracé (facultatif)
    # xlabel, ylabel, zlabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    # cmap : nom de la carte de couleur
    # marker : type de marker
    # s : taille du marker
    # ec : couleur du contour des marker
    # lw : taille du contour des marker
    
    if fig == "": fig = plt.figure()
    if minmax[0] < minmax[-1]:
      p=plt.scatter(x_obs, y_obs, marker = marker, c = z_obs, s = s, cmap=cmap, vmin = minmax[0], \
      vmax = minmax[-1], edgecolor = ec, linewidth=lw)
    else:
      p=plt.scatter(x_obs, y_obs, marker = marker, c = z_obs, s = s, cmap=cmap, edgecolor = ec, linewidth=lw)
    dx = max(x_obs)-min(x_obs)
    dy = max(y_obs)-min(y_obs)
    minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
    miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    plt.xlim([minx,maxx])
    plt.ylim([miny,maxy])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if cb:
      fig.colorbar(p,ax=plt.gca(),label=zlabel,fraction=0.046, pad=0.04)
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()

def plot_triangulation(x_obs, y_obs, xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé de la triangulation sur des sites d'observations
    # x_obs, y_obs : observations
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    from scipy.spatial import Delaunay as delaunay
    tri = delaunay(np.hstack((x_obs,y_obs)))
    
    plt.figure()
    plt.triplot(x_obs[:,0], y_obs[:,0], tri.simplices)
    plt.plot(x_obs, y_obs, 'or', ms=4)
    dx = max(x_obs)-min(x_obs)
    dy = max(y_obs)-min(y_obs)
    minx = min(x_obs)-0.05*dx; maxx = max(x_obs)+0.05*dx
    miny = min(y_obs)-0.05*dy; maxy = max(y_obs)+0.05*dy
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
    return plt.gca()