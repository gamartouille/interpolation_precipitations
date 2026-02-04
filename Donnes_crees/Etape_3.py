
from Etape_1 import charger_contours, charger_precipitations
from interpolateurs import interp_lin, interp_ppv, interp_inv, interp_splines, interp_krg
import numpy as np
import matplotlib.pyplot as plt

def wilmott(ind_test, z_obs, err):

    z_obs_extrait = z_obs[ind_test]
    z_pred = z_obs_extrait + err
    z_obs_moy = np.nanmean(z_obs_extrait)

    num = np.nansum((z_pred - z_obs_extrait)**2)
    denom = np.nansum((np.abs(z_pred - z_obs_moy) + np.abs(z_obs_extrait - z_obs_moy))**2)

    return 1 - (num/denom)







if __name__ == '__main__':
    FR_contours = charger_contours("/Users/clarabouvier/Desktop/interpol/interpolation_precipitations/Donnees_sources/FR_contour.txt")
    x_obs, y_obs, z_obs = charger_precipitations('/Users/clarabouvier/Desktop/interpol/interpolation_precipitations/Donnees_sources/FR_precipitation_2025.txt')
    

    # extraction de points du jeu de données
    n_test = 100
    ind_test = np.random.choice(len(z_obs), size=n_test, replace=False)



    # calcul de l'interpolation d'un point du jeu de donnée avec les 4 méthode ainsi que son erreur
    
    err_lin = []
    err_spl = []
    err_inv = []
    err_krg = []
    
    #tab_pred = []
    for i in ind_test:

        # point retiré
        x0, y0, z_obs_ = x_obs[i], y_obs[i], z_obs[i]

        # jeu de données sans le point retiré
        mask = np.ones(len(z_obs), dtype=bool)
        mask[i] = False

        X_train = x_obs[mask]
        Y_train = y_obs[mask]
        Z_train = z_obs[mask]

        # interpolations
        z_pred_lin = interp_lin(X_train, Y_train, Z_train, x0, y0)
        z_pred_inv = interp_inv(X_train, Y_train, Z_train, x0, y0)

        err_lin.append(z_pred_lin - z_obs_)
        err_inv.append(z_pred_inv - z_obs_)

    err_lin = np.array(err_lin).reshape((n_test,1))
    err_inv = np.array(err_inv).reshape((n_test,1))


    err_lin_moy = np.nanmean(err_lin)
    err_lin_std = np.nanstd(err_lin)
    err_lin_rmse = np.sqrt(np.nanmean(err_lin**2))
    err_lin_wilmott = wilmott(ind_test, z_obs, err_lin)

    err_inv_moy = np.nanmean(err_inv)
    err_inv_std = np.nanstd(err_inv)
    err_inv_rmse = np.sqrt(np.nanmean(err_inv**2))
    err_inv_wilmott = wilmott(ind_test, z_obs, err_inv)

    print(f"\n\n--- Interpolation linéaire ---\n\n")

    print(f"erreur moyenne : {err_lin_moy} mm")
    print(f"écart-type de l'erreur :{err_lin_std} mm ")
    print(f"erreur moyenne quadratique : {err_lin_rmse}")
    print(f"indice de Wilmott : {err_lin_wilmott}")

    plt.figure()
    plt.hist(err_lin, bins=20)
    plt.xlabel('erreur sur les précipitations [mm]')
    plt.ylabel("nb d'occurences")
    plt.title(f"histogramme des erreurs de précipitations \n par interpolation linéaire")
    plt.show()

    print(err_lin)

    print(f"\n\n--- Interpolation par inverse des distances ---\n\n")

    print(f"erreur moyenne : {err_inv_moy} mm")
    print(f"écart-type de l'erreur :{err_inv_std} mm ")
    print(f"erreur moyenne quadratique : {err_inv_rmse}")
    print(f"indice de Wilmott : {err_inv_wilmott}")









    













