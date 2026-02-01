import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Etape_1 import charger_contours, charger_precipitations
from interpolateurs import interp_lin, interp_ppv, interp_inv, interp_splines

def carte_France(x,y):
    plt.plot(x, y, color='black')
    plt.title("Contours de la France")
    plt.xlabel("Coordonnée Est (m)")
    plt.ylabel("Coordonnée Nord (m)")
    plt.axis('equal')
    plt.show()

def carte_precipitations(x, y, z):
    plt.scatter(x, y, c=z, cmap='Blues')
    plt.colorbar(label='Précipitations (mm)')
    plt.title("Carte des précipitations")
    plt.xlabel("Coordonnée Est (m)")
    plt.ylabel("Coordonnée Nord (m)")
    plt.axis('equal')
    plt.show()

def interpole_30km(x_obs, y_obs, z_obs, step, methode):

    interpoles = []
    x_obs = np.sort(x_obs)
    y_obs = np.sort(y_obs)
    z_obs = np.sort(z_obs)

    for i in range(x_obs.shape[0]):
        x_a_interpoler = []
        y_a_interpoler = []
        z_a_interpoler = []

        while len(x_a_interpoler) <= step:
            print(x_obs[i:i+step])
            x_a_interpoler.extend(x_obs[i:i+step]).astype(float)
            y_a_interpoler.extend(y_obs[i:i+step])
            z_a_interpoler.extend(z_obs[i:i+step])
        
        print(x_a_interpoler)
        print(y_a_interpoler)
        print(z_a_interpoler)

        if methode == 'lin':
            z_interpole = interp_lin(x_a_interpoler[:0], y_a_interpoler[:0], z_a_interpoler[:0], np.array([x_obs[i]]), np.array([y_obs[i]]))
            interpoles.append(z_interpole)

        elif methode == 'ppv':
            z_interpole = interp_ppv(np.array(x_a_interpoler), np.array(y_a_interpoler), np.array(z_a_interpoler), np.array([x_obs[i]]), np.array([y_obs[i]]))
            interpoles.append(z_interpole)

        elif methode == 'inv':
            z_interpole = interp_inv(np.array(x_a_interpoler), np.array(y_a_interpoler), np.array(z_a_interpoler), np.array([x_obs[i]]), np.array([y_obs[i]]), p=2)
            interpoles.append(z_interpole)

        elif methode == 'splines':
            z_interpole = interp_splines(np.array(x_a_interpoler), np.array(y_a_interpoler), np.array(z_a_interpoler), np.array([x_obs[i]]), np.array([y_obs[i]]))
            interpoles.append(z_interpole)
        
    return np.array(interpoles)


if __name__ == '__main__':
    FR_contours = charger_contours("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_contour.txt")
    FR_precipitations = charger_precipitations("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_precipitation_2025.txt")
    x_obs, y_obs, z_obs = charger_precipitations("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_precipitation_2025.txt")
    carte_France(FR_contours[0], FR_contours[1])
    carte_precipitations(x_obs, y_obs, interpole_30km(x_obs, y_obs, z_obs, 30, 'lin'))