from interpolateurs import interp_lin, interp_inv, interp_splines, moindres_carres, interp_krg_pt
import numpy as np
import matplotlib.pyplot as plt

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

    obs = np.loadtxt(donnees)
    E = obs[:,0:1]/1e3 #conversion en km
    N = obs[:,1:2]/1e3 #conversion en km
    TP = obs[:,2:3]

    return E, N, TP

def charger_villes(donnees):
    """    
    donnees: Fichier texte contenant les coordonnées des villes à étudier
    """

    villes = np.loadtxt(donnees, skiprows=1, dtype=str)
    x = villes[:,1].astype(float)
    y = villes[:,2].astype(float)

    return villes[:,0], x, y

if __name__ == "__main__":

    #######INTERPOLATION VILLES###############

    villes = charger_villes("Donnees_sources/FR_coords_villes.txt") 
    x_obs, y_obs, z_obs = charger_precipitations("Donnees_sources/FR_precipitation_2025.txt")

    # kriegage par fitting cubique

    a0, c0 = moindres_carres(x_obs, y_obs, z_obs, hmax=800, nbin=20)


    for i in range(len(villes[0])):

        x = np.array([villes[1][i]])
        y = np.array([villes[2][i]])

        krg_z, krg_inc = interp_krg_pt(x_obs, y_obs, z_obs, x, y, c0, a0)

        print(f"\n\n--- Interpolations pour la ville de {villes[0][i]} ---\n\n")
        print(f"Interpolation Linéaire {villes[0][i]} : ", interp_lin(x_obs, y_obs, z_obs, x, y)[0], ' mm')
        print(f"Interpolation Inverse des Distances {villes[0][i]} : ", interp_inv(x_obs, y_obs, z_obs, x, y, p=2)[0], ' mm')
        print(f"Interpolation par Splines {villes[0][i]} : ", interp_splines(x_obs, y_obs, z_obs, x, y)[0], ' mm')
        print(f"Interpolation par Krigeage {villes[0][i]} : ", krg_z, ' mm')
        print(f"Incertitude interpolation par Kriegage {villes[0][i]} :", np.sqrt(krg_inc), ' mm')


    
    

    

    