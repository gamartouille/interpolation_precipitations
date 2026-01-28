from interpolateurs import interp_lin, interp_ppv, interp_inv, interp_splines
import numpy as np

#######CHARGEMENT DES DONNEES##########

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


if __name__ == "__main__":

    #######INTERPOLATION BREST###############

    x_Brest = np.array([145.7])
    y_Brest = np.array([6835.2])

    x_obs, y_obs, z_obs = charger_precipitations("D:/ENSG/Geostatistiques/interpolation_precipitations/Donnees_sources/FR_precipitation_2025.txt")

    print("Interpolation Linéaire Brest : ", interp_lin(x_obs, y_obs, z_obs, x_Brest, y_Brest))
    print("Interpolation Par Voisin le Plus Proche Brest : ", interp_ppv(x_obs, y_obs, z_obs, x_Brest, y_Brest))
    print("Interpolation Inverse des Distances Brest : ", interp_inv(x_obs, y_obs, z_obs, x_Brest, y_Brest, p=2))
    print("Interpolation par Splines Brest : ", interp_splines(x_obs, y_obs, z_obs, x_Brest, y_Brest))