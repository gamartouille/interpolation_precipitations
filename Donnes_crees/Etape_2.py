import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Etape_1 import charger_contours, charger_precipitations
from interpolateurs import interp_lin, interp_inv, interp_splines, interp_krg

def carte_France(x,y):
    plt.plot(x, y, color='black')
    plt.title("Contours de la France")
    plt.xlabel("Coordonnée Est (m)")
    plt.ylabel("Coordonnée Nord (m)")
    plt.axis('equal')
    plt.show()

def carte_precipitations(x, y, z, x_france, y_france, methode):
    plt.scatter(x, y, c=z, cmap='Blues')
    plt.plot(x_france, y_france, color='black')
    plt.colorbar(label='Précipitations (mm)')
    plt.title(f"Carte des précipitations suivant la méthode {methode}")
    plt.xlabel("Coordonnée Est (m)")
    plt.ylabel("Coordonnée Nord (m)")
    plt.axis('equal')
    plt.show()

def interpole_30km(x_obs, y_obs, z_obs, step, methode):
    '''
    Processus :
    - on crée la grille d'interpolation avec un point tous les 30km
    - on interpole nos observations avec cette grille théorique
    - on ressort les z interpolés à caler dans la carte
    '''
    
    x_reg = np.arange(np.floor(np.min(x_obs)), np.ceil(np.max(x_obs)), step)
    y_reg = np.arange(np.floor(np.min(y_obs)), np.ceil(np.max(y_obs)), step)
    x_30, y_30 = np.meshgrid(x_reg, y_reg)
    
    # Aplatir la grille en vecteurs 1D
    x_flat = x_30.flatten()
    y_flat = y_30.flatten()
    
    # Interpolation
    if methode == 'lin':
        z_interpole = interp_lin(x_obs, y_obs, z_obs, x_flat, y_flat)
        z_grid = z_interpole.reshape(x_30.shape)

    elif methode == 'inv':
        z_interpole = interp_inv(x_obs, y_obs, z_obs, x_flat, y_flat)
        z_grid = z_interpole.reshape(x_30.shape)

    elif methode == 'splines':
        z_interpole = interp_splines(x_obs, y_obs, z_obs, x_flat, y_flat)
        z_grid = z_interpole.reshape(x_30.shape)

    elif methode == 'kri':
        z_interpole = interp_krg(x_obs, y_obs, z_obs, x_flat, y_flat)[0]

    
    return x_30, y_30, z_grid


if __name__ == '__main__':
    FR_contours = charger_contours("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_contour.txt")
    x_obs, y_obs, z_obs = charger_precipitations("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_precipitation_2025.txt")
    
    x_grid_splines, y_grid_splines, z_grid_splines = interpole_30km(x_obs, y_obs, z_obs, 30, 'splines')
    x_grid_lin, y_grid_lin, z_grid_lin = interpole_30km(x_obs, y_obs, z_obs, 30, 'lin')
    x_grid_inv, y_grid_inv, z_grid_inv = interpole_30km(x_obs, y_obs, z_obs, 30, 'inv')

    carte_precipitations(x_grid_splines, y_grid_splines, z_grid_splines, FR_contours[0], FR_contours[1], 'Splines')
    carte_precipitations(x_grid_lin, y_grid_lin, z_grid_lin, FR_contours[0], FR_contours[1], 'Linéaire')
    carte_precipitations(x_grid_inv, y_grid_inv, z_grid_inv, FR_contours[0], FR_contours[1], 'Inverse des distances')