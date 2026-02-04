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

    '''
    Processus :
    - on crée la grille d'interpolation avec un point tous les 30km
    - on interpole nos observations avec cette grille théorique
    - on ressort les z interpolés à caler dans la carte
    '''
    interpoles = []

    x_min, x_max = np.min(x_obs), np.max(x_obs)
    y_min, y_max = np.min(y_obs), np.max(y_obs)
    x_30 = np.arange(x_min, x_max, step)
    y_30 = np.arange(y_min, y_max, step)

    # Interpolation
    if methode == 'lin':
        #renvoie une valeur unique : le z interpolé sur la zone entrée
        z_interpole = interp_lin(x_obs, y_obs, z_obs, x_30, y_30)
        interpoles.append(z_interpole)

    elif methode == 'ppv':
        z_interpole = interp_ppv(np.array(x_obs), np.array(y_obs), np.array(z_obs), np.array([x_30[i]]), np.array([y_30[i]]))
        interpoles.append(z_interpole)

    elif methode == 'inv':
        z_interpole = interp_inv(np.array(x_obs), np.array(y_obs), np.array(z_obs), np.array([x_30[i]]), np.array([y_30[i]]), p=2)
        interpoles.append(z_interpole)

    elif methode == 'splines':
        z_interpole = interp_splines(np.array(x_obs), np.array(y_obs), np.array(z_obs), np.array([x_30[i]]), np.array([y_30[i]]))
        interpoles.append(z_interpole)
    
    x_30_final = x_30.tolist()
    x_30_final.remove(x_30_final[0])
    print(interpoles[0].tolist()[0])
    interpoles.remove(interpoles[0].tolist()[0])

    print(len(x_30_final), len(y_30.tolist()), len(interpoles))
    return x_30_final, y_30.tolist(), interpoles


if __name__ == '__main__':
    FR_contours = charger_contours("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_contour.txt")
    FR_precipitations = charger_precipitations("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_precipitation_2025.txt")
    x_obs, y_obs, z_obs = charger_precipitations("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_precipitation_2025.txt")
    #carte_France(FR_contours[0], FR_contours[1])
    #print(interpole_30km(x_obs, y_obs, z_obs, 30, 'lin'))
    carte_precipitations(interpole_30km(x_obs, y_obs, z_obs, 30, 'lin')[0], interpole_30km(x_obs, y_obs, z_obs, 30, 'lin')[1], interpole_30km(x_obs, y_obs, z_obs, 30, 'lin')[2])