import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def charger_contours(donnees):
    """    
    donnees: Fichier texte contenant les coordonnées des contours de la France
    """

    FR_contour =np.loadtxt(donnees)
    Ec = FR_contour[:,0:1]/1e3 #conversion en km
    Nc = FR_contour[:,1:2]/1e3 #conversion en km

    return Ec, Nc

def charger_precipitations(donnees):
    """    
    donnees: Fichier texte contenant les coordonnées des points et les précipitations associées
    """

    obs =np.loadtxt(donnees)
    E = obs[:,0:1]/1e3 #conversion en km
    N = obs[:,1:2]/1e3 #conversion en km
    TP = obs[:,2:3]

    return E, N, TP

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

if __name__ == '__main__':
    contours = charger_contours("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_contour.txt")
    precipitations = charger_precipitations("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_precipitation_2025.txt")
    carte_France(contours[0], contours[1])
    carte_precipitations(precipitations[0], precipitations[1], precipitations[2])