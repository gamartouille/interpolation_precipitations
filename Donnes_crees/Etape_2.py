import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.path as pth
import pandas as pd
import numpy as np
from Etape_1 import charger_contours, charger_precipitations
from interpolateurs import interp_lin, interp_inv, interp_splines, interp_krg
import os

def is_in(FR_contours, x_test, y_test):
    # FR_contours est un tuple (x_array, y_array)
    # Il faut le convertir en array (N, 2)
    x_contour, y_contour = FR_contours
    vertices = np.column_stack((x_contour.flatten(), y_contour.flatten()))
    
    poly = pth.Path(vertices)
    points = np.array([[x_test, y_test]])
    
    return poly.contains_points(points)[0]  # Retourne un booléen, pas un array



def carte_France(x,y):
    plt.plot(x, y, color='black')
    plt.title("Contours de la France")
    plt.xlabel("Coordonnée Est (m)")
    plt.ylabel("Coordonnée Nord (m)")
    plt.axis('equal')
    plt.show()

def carte_precipitations(x, y, z, x_france, y_france, methode):
    path = os.path.dirname(os.path.abspath(__file__))
    NCL = col.ListedColormap(np.loadtxt(os.path.join(path, "ncl.rgb"))/255)
    plt.scatter(x, y, c=z, marker='s', cmap=NCL, vmin=0, vmax=2000)
    plt.plot(x_france, y_france, color='black')
    plt.colorbar(label='Précipitations (mm)')
    plt.title(f"Carte des précipitations suivant la méthode {methode}")
    plt.xlabel("Coordonnée Est (m)")
    plt.ylabel("Coordonnée Nord (m)")
    plt.axis('equal')
    plt.show()

def interpole_30km(x_obs, y_obs, z_obs, step, methode, FR_contours):
    '''
    Processus :
    - on crée la grille d'interpolation avec un point tous les 30km
    - on interpole nos observations avec cette grille théorique
    - on garde seulement les points dans les contours de la France
    '''
    
    x_reg = np.arange(np.floor(np.min(x_obs)), np.ceil(np.max(x_obs)), step)
    y_reg = np.arange(np.floor(np.min(y_obs)), np.ceil(np.max(y_obs)), step)
    
    # Créer la grille complète d'abord
    x_30, y_30 = np.meshgrid(x_reg, y_reg)
    
    # Préparer le polygone une seule fois
    x_contour, y_contour = FR_contours
    vertices = np.column_stack((x_contour.flatten(), y_contour.flatten()))
    poly = pth.Path(vertices)
    
    # Tester tous les points de la grille
    points_grille = np.column_stack((x_30.flatten(), y_30.flatten()))
    masque_france = poly.contains_points(points_grille)
    masque_france = masque_france.reshape(x_30.shape)
    
    # Aplatir la grille
    x_flat = x_30.flatten()
    y_flat = y_30.flatten()
    
    # Interpolation
    if methode == 'lin':
        z_interpole = interp_lin(x_obs, y_obs, z_obs, x_flat, y_flat)
    elif methode == 'inv':
        z_interpole = interp_inv(x_obs, y_obs, z_obs, x_flat, y_flat)
    elif methode == 'splines':
        z_interpole = interp_splines(x_obs, y_obs, z_obs, x_flat, y_flat)
    elif methode == 'kri':
        a0 = 578.6810308
        c0 = 67715.67450193148
        z_interpole = interp_krg(x_obs, y_obs, z_obs, x_flat, y_flat, c0, a0)[0]
    
    # Remettre en forme de grille
    z_grid = z_interpole.reshape(x_30.shape)
    
    # Mettre NaN pour les points hors de France
    z_grid[~masque_france] = np.nan
    
    return x_30, y_30, z_grid

if __name__ == '__main__':
    FR_contours = charger_contours("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_contour.txt")
    x_obs, y_obs, z_obs = charger_precipitations("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_precipitation_2025.txt")

    x_grid_splines, y_grid_splines, z_grid_splines = interpole_30km(x_obs, y_obs, z_obs, 30, 'splines', FR_contours)
    x_grid_lin, y_grid_lin, z_grid_lin = interpole_30km(x_obs, y_obs, z_obs, 30, 'lin', FR_contours)
    x_grid_inv, y_grid_inv, z_grid_inv = interpole_30km(x_obs, y_obs, z_obs, 30, 'inv', FR_contours)
    x_grid_kri, y_grid_kri, z_grid_kri = interpole_30km(x_obs, y_obs, z_obs, 30, 'kri', FR_contours)

    carte_precipitations(x_grid_splines, y_grid_splines, z_grid_splines, FR_contours[0], FR_contours[1], 'Splines')
    carte_precipitations(x_grid_lin, y_grid_lin, z_grid_lin, FR_contours[0], FR_contours[1], 'Linéaire')
    carte_precipitations(x_grid_inv, y_grid_inv, z_grid_inv, FR_contours[0], FR_contours[1], 'Inverse des distances')
    carte_precipitations(x_grid_kri, y_grid_kri, z_grid_kri, FR_contours[0], FR_contours[1], 'Krigeage')