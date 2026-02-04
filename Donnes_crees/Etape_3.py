
from Etape_1 import charger_contours, charger_precipitations
from interpolateurs import interp_lin, interp_inv, interp_splines, interp_krg_pt
import numpy as np
import matplotlib.pyplot as plt

def wilmott(ind_test, z_obs, err):
    """Calcule l'indice de Wilmott.

    Args:
        ind_test (array): indices des points retirés du jeu de données
        z_obs (array): vecteur des observations 
        err (array): vecteur des erreurs 

    Returns:
        float: indice de Wilomott
    """

    # récupération des observations correspondant aux points retirés
    z_obs_extrait = z_obs[ind_test]
    z_pred = z_obs_extrait + err
    z_obs_moy = np.nanmean(z_obs_extrait)

    num = np.nansum((z_pred - z_obs_extrait)**2)
    denom = np.nansum((np.abs(z_pred - z_obs_moy) + np.abs(z_obs_extrait - z_obs_moy))**2)

    return 1 - (num/denom)


if __name__ == '__main__':
    x_obs, y_obs, z_obs = charger_precipitations('Donnees_sources/FR_precipitation_2025.txt')
    
    a0 = 578.6810308
    c0 = 67715.67450193148

    # extraction de points du jeu de données
    n_test = 100
    ind_test = np.random.choice(len(z_obs), size=n_test, replace=False)

    # calcul de l'interpolation d'un point du jeu de donnée avec les 4 méthode ainsi que son erreur
    
    err_lin = []
    err_spl = []
    err_inv = []
    err_krg = []

    z_pred_lin_tab = []
    z_pred_inv_tab = []
    z_pred_spl_tab = []
    z_pred_krg_tab = []
    
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
        z_pred_spl = interp_splines(X_train, Y_train, Z_train, x0, y0)
        z_pred_krg = interp_krg_pt(X_train, Y_train, Z_train, x0, y0, c0, a0)[0]


        z_pred_lin_tab.append(z_pred_lin)
        z_pred_inv_tab.append(z_pred_inv)
        z_pred_spl_tab.append(z_pred_spl)
        z_pred_krg_tab.append(z_pred_krg)

        # calcul de l'erreur
        err_lin.append(z_pred_lin - z_obs_)
        err_inv.append(z_pred_inv - z_obs_)
        err_spl.append(z_pred_spl - z_obs_)
        err_krg.append(z_pred_krg - z_obs_)


    err_lin = np.array(err_lin).reshape((n_test,1))
    err_inv = np.array(err_inv).reshape((n_test,1))
    err_spl = np.array(err_spl).reshape((n_test,1))
    err_krg = np.array(err_krg).reshape((n_test,1))

    z_pred_lin_tab = np.array(z_pred_lin_tab).reshape((n_test,1))
    z_pred_inv_tab = np.array(z_pred_inv_tab).reshape((n_test,1))
    z_pred_spl_tab = np.array(z_pred_spl_tab).reshape((n_test,1))
    z_pred_krg_tab = np.array(z_pred_krg_tab).reshape((n_test,1))

    # calculs des statistiques

    err_lin_moy = np.nanmean(err_lin)
    err_lin_std = np.nanstd(err_lin)
    err_lin_rmse = np.sqrt(np.nanmean(err_lin**2))
    err_lin_wilmott = wilmott(ind_test, z_obs, err_lin)

    err_inv_moy = np.nanmean(err_inv)
    err_inv_std = np.nanstd(err_inv)
    err_inv_rmse = np.sqrt(np.nanmean(err_inv**2))
    err_inv_wilmott = wilmott(ind_test, z_obs, err_inv)

    err_spl_moy = np.nanmean(err_spl)
    err_spl_std = np.nanstd(err_spl)
    err_spl_rmse = np.sqrt(np.nanmean(err_spl**2))
    err_spl_wilmott = wilmott(ind_test, z_obs, err_spl)

    err_krg_moy = np.nanmean(err_krg)
    err_krg_std = np.nanstd(err_krg)
    err_krg_rmse = np.sqrt(np.nanmean(err_krg**2))
    err_krg_wilmott = wilmott(ind_test, z_obs, err_krg)

###################### LINEAIRE ##############################

    print(f"\n\n--- Interpolation linéaire ---\n\n")

    print(f"erreur moyenne : {err_lin_moy} mm")
    print(f"écart-type de l'erreur :{err_lin_std} mm ")
    print(f"erreur moyenne quadratique : {err_lin_rmse}")
    print(f"indice de Wilmott : {err_lin_wilmott}")

    # diagramme de corrélation et histogramme des erreurs

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 5))

    ax[0].hist(err_lin, bins=50, edgecolor='black', alpha=0.7)
    ax[0].set_xlabel("erreur d'interpolation [mm]")
    ax[0].set_ylabel("nb d'occurences")
    ax[0].set_title(f"histogramme des erreurs d'interpolation linéaire")
    ax[0].grid(True, linestyle=':', alpha=0.6)

    vmin = min(np.nanmin(z_obs[ind_test]), np.nanmin(z_pred_lin_tab))
    vmax = max(np.nanmax(z_obs[ind_test]), np.nanmax(z_pred_lin_tab))
    ax[1].scatter(z_obs[ind_test], z_pred_lin_tab, marker='+')
    ax[1].plot([vmin, vmax], [vmin, vmax], linestyle='--', linewidth=1, color='r')
    ax[1].set_xlabel('précipitation observée [mm]')
    ax[1].set_ylabel('précipitation prédite [mm]')
    ax[1].set_title(f"diagramme de corrélation \n de l'interpolation linéaire")
    ax[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

###################### INVERSE ##############################

    print(f"\n\n--- Interpolation par inverse des distances ---\n\n")

    print(f"erreur moyenne : {err_inv_moy} mm")
    print(f"écart-type de l'erreur :{err_inv_std} mm ")
    print(f"erreur moyenne quadratique : {err_inv_rmse}")
    print(f"indice de Wilmott : {err_inv_wilmott}")

    # diagramme de corrélation et histogramme des erreurs

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 5))

    ax[0].hist(err_inv, bins=50, edgecolor='black', alpha=0.7)
    ax[0].set_xlabel("erreur d'interpolation [mm]")
    ax[0].set_ylabel("nb d'occurences")
    ax[0].set_title(f"histogramme des erreurs d'interpolation \n par inverse des distances")
    ax[0].grid(True, linestyle=':', alpha=0.6)

    vmin = min(np.nanmin(z_obs[ind_test]), np.nanmin(z_pred_inv_tab))
    vmax = max(np.nanmax(z_obs[ind_test]), np.nanmax(z_pred_inv_tab))
    ax[1].scatter(z_obs[ind_test], z_pred_inv_tab, marker='+')
    ax[1].plot([vmin, vmax], [vmin, vmax], linestyle='--', linewidth=1, color='r')
    ax[1].set_xlabel('précipitation observée [mm]')
    ax[1].set_ylabel('précipitation prédite [mm]')
    ax[1].set_title(f"diagramme de corrélation \n de l'interpolation par inverse des distances")
    ax[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

###################### SPLINES ##############################

    print(f"\n\n--- Interpolation par splines ---\n\n")

    print(f"erreur moyenne : {err_spl_moy} mm")
    print(f"écart-type de l'erreur :{err_spl_std} mm ")
    print(f"erreur moyenne quadratique : {err_spl_rmse}")
    print(f"indice de Wilmott : {err_spl_wilmott}")

    # diagramme de corrélation et histogramme des erreurs

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 5))

    ax[0].hist(err_spl, bins=50, edgecolor='black', alpha=0.7)
    ax[0].set_xlabel("erreur d'interpolation [mm]")
    ax[0].set_ylabel("nb d'occurences")
    ax[0].set_title(f"histogramme des erreurs d'interpolation \n par splines")
    ax[0].grid(True, linestyle=':', alpha=0.6)

    vmin = min(np.nanmin(z_obs[ind_test]), np.nanmin(z_pred_spl_tab))
    vmax = max(np.nanmax(z_obs[ind_test]), np.nanmax(z_pred_spl_tab))
    ax[1].scatter(z_obs[ind_test], z_pred_spl_tab, marker='+')
    ax[1].plot([vmin, vmax], [vmin, vmax], linestyle='--', linewidth=1, color='r')
    ax[1].set_xlabel('précipitation observée [mm]')
    ax[1].set_ylabel('précipitation prédite [mm]')
    ax[1].set_title(f"diagramme de corrélation \n de l'interpolation par splines")
    ax[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

###################### KRIEGAGE ##############################

    print(f"\n\n--- Interpolation par kriegage ---\n\n")

    print(f"erreur moyenne : {err_krg_moy} mm")
    print(f"écart-type de l'erreur :{err_krg_std} mm ")
    print(f"erreur moyenne quadratique : {err_krg_rmse}")
    print(f"indice de Wilmott : {err_krg_wilmott}")

    # diagramme de corrélation et histogramme des erreurs

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 5))

    ax[0].hist(err_krg, bins=50, edgecolor='black', alpha=0.7)
    ax[0].set_xlabel("erreur d'interpolation [mm]")
    ax[0].set_ylabel("nb d'occurences")
    ax[0].set_title(f"histogramme des erreurs d'interpolation \n par kriegage")
    ax[0].grid(True, linestyle=':', alpha=0.6)

    vmin = min(np.nanmin(z_obs[ind_test]), np.nanmin(z_pred_krg_tab))
    vmax = max(np.nanmax(z_obs[ind_test]), np.nanmax(z_pred_krg_tab))
    ax[1].scatter(z_obs[ind_test], z_pred_krg_tab, marker='+')
    ax[1].plot([vmin, vmax], [vmin, vmax], linestyle='--', linewidth=1, color='r')
    ax[1].set_xlabel('précipitation observée [mm]')
    ax[1].set_ylabel('précipitation prédite [mm]')
    ax[1].set_title(f"diagramme de corrélation \n de l'interpolation par kriegage")
    ax[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()








    













