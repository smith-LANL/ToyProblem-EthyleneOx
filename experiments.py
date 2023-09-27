"""
© 2023. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are.
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare.
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit.
others to do so.

Created March 2023 @author: Sean T. Smith
"""
from numpy import array
from pandas import DataFrame

# Isolated reactions:
# Klugherz & Harriott 1971 (for more details see the jupyter notebook EthyleneOxRates-Literature):
# Reference conditions and reaction rate, as described on page 858:
expt_rxn_aux = dict(Tavg=220,   # avg. reactor temperature [degC]
                    Peth_ref=0.263,  # reference partial pressure of ethylene [atm]
                    Poxy_ref=0.263,  # reference partial pressure of oxygen [atm]
                    Pinert=0.789,  # fixed partial pressure of inert (helium in the expt.) [atm]
                    # reference rate of ethylene oxide production [mol rxn. / (min * g-cat.)]
                    Rethox_ref=(6 + 7) / 2 * 1e-6,
                    # reference rate of CO2/water production [mol rxn. / (min * g-cat.)]
                    Rco2_ref=(4 + 6) / 2 * 1e-6)
# Rate of ethylene ox. production at various pressures of ethylene & oxygen — experiment:
# As scanned from Fig. 3:
Po = array([ 0.061,  0.132,  0.263,  0.526,  0.789,  1.05,    1.32])  # [atm]
Pe = array([ 0.026,  0.066,  0.132,  0.263,  0.526,  0.789,   1.05,   1.32])  # [atm]
R0ethox = [[  None, 0.2318, 0.3146, 0.2980, 0.2252,   None, 0.1192,   None],  # Po = 0.61 atm
           [  None, 0.3907, 0.5232, 0.5960, 0.5199, 0.4305, 0.3560, 0.2930],  # Po = 0.132 atm
           [0.3195, 0.6076, 0.8162, 1.0050, 0.9652, 0.8725, 0.7202, 0.6970],  # Po = 0.263 atm
           [  None, 0.8990, 1.2070, 1.6209, 1.7997, 1.6672, 1.6076,   None],  # Po = 0.526 atm
           [0.5149, 1.0745, 1.4851, 2.1474, 2.5679, 2.3791,   None,   None],  # Po = 0.789 atm
           [  None, 1.1540, 1.8328, 2.5513, 3.1440,   None,   None,   None],  # Po = 1.05 atm
           [0.5811, 1.2467, 1.9719,   None,   None,   None,   None,   None]]  # Po = 1.32 atm
expt = []
for i in range(Pe.shape[0]):
    for j in range(Po.shape[0]):
        if R0ethox[j][i] is not None:
            expt.append([Pe[i], Po[j], 0.01, R0ethox[j][i], None])

# As scanned from Fig. 4:
#       Po = 0.061,  0.132,  0.263,  0.526,  0.789,  1.05,    1.32 atm
R0ethox = [[  None,   None, 0.3342,   None, 0.5318,   None, 0.5812],  # Pe = 0.026 atm
           [0.2590, 0.4001, 0.6240, 0.9138, 1.0851, 1.1608, 1.2530],  # Pe = 0.066 atm
           [0.3178, 0.5410, 0.8150, 1.2168, 1.4967, 1.8425, 1.9808],  # Pe = 0.132 atm
           [0.2980, 0.6043, 1.0258, 1.6317, 2.1520, 2.5670,   None],  # Pe = 0.263 atm
           [0.2289, 0.5417, 0.9731, 1.8128, 2.5735, 3.1465,   None]]  # Pe = 0.526 atm
for i, Pe_val in enumerate(Pe[:5]):
    for j in range(Po.shape[0]):
        if R0ethox[i][j] is not None:
            expt.append([Pe_val, Po[j], 0.01, R0ethox[i][j], None])

# As scanned from Fig. 5:
#       Po = 0.061,  0.132,  0.263,  0.526,  0.789,  1.05,    1.32 atm
R0ethox = [[0.2384, 0.5364, 0.9801, 1.8113, 2.5695, 3.1457,   None],  # Pe = 0.526 atm
           [  None, 0.4437, 0.8808, 1.6788, 2.3775,   None,   None],  # Pe = 0.789 atm
           [0.1291, 0.3642, 0.7384, 1.6192,   None,   None,   None],  # Pe = 1.05 atm
           [  None, 0.2947, 0.6954,   None,   None,   None,   None]]  # Pe = 1.32 atm
for i, Pe_val in enumerate(Pe[4:8]):
    for j in range(Po.shape[0]):
        if R0ethox[i][j] is not None:
            expt.append([Pe_val, Po[j], 0.01, R0ethox[i][j], None])

# Rate of CO2/water production at various pressures of ethylene & oxygen — experiment:
# As scanned from Fig. 6:
#       Pe = 0.026,  0.066,  0.132,  0.263,  0.526,  0.789,   1.05,   1.32 atm
R0co2 =   [[  None, 0.3551, 0.4041, 0.3061, 0.1959,   None, 0.1163,   None],  # Po = 0.061 atm
           [  None, 0.5571, 0.6429, 0.6000, 0.4041, 0.3000, 0.2388, 0.1837],  # Po = 0.132 atm
           [0.3735, 0.7592, 1.0041, 0.9980, 0.7531, 0.6367, 0.4776, 0.4469],  # Po = 0.263 atm
           [  None, 0.9612, 1.3224, 1.6286, 1.4571, 1.2000, 1.0592,   None],  # Po = 0.526 atm
           [0.4776, 1.0469, 1.5490, 2.0816, 2.1612, 1.7633,   None,   None],  # Po = 0.789 atm
           [  None, 1.1633, 1.8000, 2.4000, 2.6265,   None,   None,   None],  # Po = 1.05 atm
           [0.5082, 1.1327, 2.0755,   None,   None,   None,   None,   None]]  # Po = 1.32 atm
for i in range(Pe.shape[0]):
    for j in range(Po.shape[0]):
        if R0co2[j][i] is not None:
            expt.append([Pe[i], Po[j], 0.01, None, R0co2[j][i]])

# Combine into a single DataFrame and average over the redundant scans of the same results:
labels = ['Pres. C2H4 [atm]', 'Pres. O2 [atm]', 'Pres. Prod. [atm]',
          'rel. rate C2H4O [-]', 'rel. rate CO2 [-]']
expt_rxn = DataFrame(expt, columns=labels).groupby(labels[:-2]).mean().reset_index()

# # As scanned from Fig. 2:
# # (This data was not used by the authors to determine parameters for their model.
# #  In deed their model is not a function of Pp.)
# # The following data was collected at Pe = 0.263 [atm]
# Po = array([1.05, 0.789, 0.526, 0.263, 0.132, 0.061])  # [atm]
# Pp = [[0.00353, 0.00476, 0.00650, 0.00675, 0.00953, 0.00972, 0.01021, 0.01188, 0.01207],  # Po = 0.061 atm
#       [0.00433, 0.00439, 0.00600, 0.00965, 0.01027, 0.01250, 0.01658, 0.02011],  # Po = 0.132 atm
#       [0.00322, 0.00340, 0.00347, 0.00359, 0.00384, 0.00644, 0.00699, 0.01405, 0.01597, 0.02314, 0.025612],  # Po = 0.263 atm
#       [0.00415, 0.00736, 0.00959, 0.01108, 0.01108, 0.01114, 0.01182, 0.01498, 0.01850, 0.02054],  # Po = 0.526 atm
#       [0.00575, 0.00699, 0.00699, 0.00761, 0.00780, 0.00823, 0.00910, 0.00910, 0.00873, 0.00953, 0.01040, 0.01176, 0.01386, 0.01832],  # Po = 0.789 atm
#       [0.00817, 0.00953, 0.00959, 0.01108, 0.01312, 0.01485]]  # Po = 1.05 atm
# R  = [[0.33610, 0.30498, 0.34855, 0.31743, 0.30498, 0.30498, 0.28008, 0.26141, 0.2739],  # Po = 0.061 atm
#       [0.81535, 0.72199, 0.70954, 0.59129, 0.59129, 0.54149, 0.53527, 0.46680],  # Po = 0.263 atm
#       [1.17635, 1.30083, 1.14523, 1.28838, 1.26971, 1.08299, 1.20747, 0.85892, 0.8714, 0.7407, 0.7282],  # Po = 0.263 atm
#       [2.12863, 1.78008, 1.71784, 1.59336, 1.56846, 1.56224, 1.54979, 1.46888, 1.3755, 1.1888],  # Po = 0.526 atm
#       [2.37759, 2.42116, 2.39627, 2.34025, 2.29668, 2.30913, 2.41494, 2.44606, 2.1224, 2.1909, 2.0788, 2.0290, 1.9170, 1.6867],  # Po = 0.789 atm
#       [2.74481, 2.65768, 2.54564, 2.43983, 2.23444, 2.28423]]  # Po = 1.05 atm


# Turbulent Nusselt Number:
expt_Nu = dict()
# Hilpert et al 1933 (as reproted by Finlayson 1987):
expt_Nu['Hilpert'] = dict(x=array([100.000000,  78.987974,  70.596064,  67.494486,  61.005059,
                                    58.324854,  50.970286,  49.838042,  45.046239,  43.067171,
                                    39.810717,  36.389459,  34.017885,  30.747150,  29.396300,
                                    25.118864,  22.703748,  21.951460,  21.706278,  18.969189,
                                    17.145349,  15.152622,  13.850435,  10.940178,   9.668650,
                                     9.038524,   7.637071,   7.301542,   6.452917,   5.898365,
                                     5.212825,   5.513957,   4.818666,   4.454310,   4.306717,
                                     3.680049,   3.479071,   2.906796,   2.146383]),
                          y=array([  5.274997,   4.641588,   4.470198,   4.370416,   4.115087,
                                     4.023232,   3.816799,   3.759788,   3.620958,   3.513594,
                                     3.435165,   3.283521,   3.186162,   3.115043,   2.977530,
                                     2.803576,   2.700054,   2.659724,   2.659724,   2.542311,
                                     2.448436,   2.305393,   2.237037,   2.043889,   1.953663,
                                     1.924481,   1.771601,   1.745139,   1.693394,   1.606506,
                                     1.547185,   1.570646,   1.501310,   1.467799,   1.435035,
                                     1.361403,   1.341068,   1.272257,   1.145047]),
                          label='Hilpert et al 1933')

# Wilke & Hougen 1945 (as reported by Wakao et al 1979):
expt_Nu['Wilke'] = dict(x=array([ 45.056040,  64.017011,  68.065929,  67.311174,  92.491472,
                                 130.684102, 155.339997, 130.684102, 189.867382, 201.876025,
                                 248.126485, 228.219900, 252.311500]),
                        y=array([ 27.564990,  29.121565,  23.700171,  24.867163,  29.121565,
                                  23.863470,  19.554759,  29.727681,  32.727327,  31.840663,
                                  28.332590,  38.856458,  39.938490]),
                        label='Wilke & Hougen 1945')

# Thoenes & Kramers 1958:
Pr = 6.9  # 0.71 for gases, 6.9 for water
expt_Nu['Thoenes'] = dict(x=array([  7.532818,  31.704329,   6.388062,  33.380276,   5.974290,
                                    17.990099,  29.957810, 258.007962,   7.008701,  25.405146,
                                    47.872872, 597.429073,  19.535644,  94.491229, 235.160650,
                                  1509.979034, 17.2638027,   4.122992,  16.566828, 160.626262,
                                    25.015563,  70.448981, 139.051837, 287.483534,2758.772471,
                                   679.539592,3089.817907, 504.035180,2845.369700,  50.403518,
                                   201.488637,  40.807337, 737.919434,  30.739425, 284.536970,
                                     4.454198, 202.529222,]),
                          y=array([  6.736626,  10.424582,   5.387372,   7.791980,   4.491280,
                                     6.071495,   7.320821,  20.382388,   3.339658,   5.331658,
                                     6.986244,  28.873312,   4.353376,   8.556179,  14.165813,
                                    54.720364,   4.880761,  2.309038,    2.561992,   7.474620,
                                     3.686306,   5.946567,   8.250467,  12.635142,  60.400199,
                                    25.487106,  58.850708,  15.474486,  41.328802,   4.755551,
                                     9.592709,   7.671421,  32.036372,   3.137718,  14.239635,
                                     1.174836,   9.896584]) * Pr**(1/3),
                          label='Thoenes & Kramers 1958')

# Collis & Williams 1959 (as reported by Finlayson 1987):
expt_Nu['Collis'] = dict(x=array([156.719108, 105.776756,  64.529172,  32.523338,  16.764486,
                                   10.342705,   6.452917,   5.832485,   5.097028,   1.793323,
                                    1.309402,   1.196874,   0.713934,   0.504009,   0.465899,
                                    0.440455,   0.430671,   0.328907,   0.248383,   0.217062,
                                    0.185478,   0.111887,   0.085449,   0.076370,   0.059649,
                                    0.052128,   0.046069,   0.040715,   0.039810,   0.035183,
                                    0.035581,   0.022703,   0.019183,   0.016953,   0.012518,
                                    0.012378,   0.010577]),
                         y=array([  6.271704,   5.157251,   4.053620,   2.889244,   2.288111,
                                    1.910054,   1.524075,   1.478885,   1.424278,   1.038340,
                                    0.920560,   0.879922,   0.723564,   0.661091,   0.656135,
                                    0.656135,   0.622468,   0.577348,   0.547724,   0.523545,
                                    0.496682,   0.453798,   0.433765,   0.433765,   0.408423,
                                    0.408423,   0.378818,   0.384562,   0.375978,   0.370362,
                                    0.381679,   0.356687,   0.333331,   0.325890,   0.316227,
                                    0.328352,   0.318616]),
                         label='Collis & Williams 1959')

# Satterfield & Resnick 1954 (as reported by Wakao et al 1979):
expt_Nu['Satterfield'] = dict(x=array([ 15.620847,  19.632617,  22.071185,  24.130514,  28.207409,
                                        27.431948,  27.894628,  31.711052,  35.649883,  40.077956,
                                        42.140272,  42.140272,  45.815976,  50.652453,  59.874198,
                                        69.213961,  74.416750,  81.360127,  79.565792, 104.561176,
                                       104.561176, 111.795970, 127.091417, 133.631237, 138.177024,
                                       149.394338, 152.763419, 164.246592]),
                              y=array([ 17.884841,  14.655612,  16.583718,  14.258556,  16.813035,
                                        15.589886,  14.756592,  15.805460,  14.858268,  19.825159,
                                        17.520188,  16.697983,  21.380599,  16.245593,  22.127401,
                                        19.420945,  20.237787,  20.801346,  22.587945,  29.727681,
                                        34.103832,  33.870458,  35.783097,  32.280951,  28.527807,
                                        29.727681,  32.727327,  29.524253]),
                              label='Satterfield & Resnick 1954')

# Galloway et al 1957 (as reported by Wakao et al 1979):
expt_Nu['Galloway'] = dict(x=array([ 154.476345,  157.960032,  197.423804,  215.844226,  250.908710,
                                     250.908710,  294.939986,  387.594605,  407.539296,  364.538448,
                                     394.131954,  473.745168,  553.785275,  515.067807,  479.057241,
                                     479.057241,  430.906025,  517.947467,  506.524532,  515.067807,
                                     619.109623,  626.051657,  615.667526,  612.244566,  669.369407,
                                     707.748462,  696.009251,  855.467253,  940.514764,  855.467253,
                                     841.277874,  748.328024,  727.755459,  822.724134,  836.600577,
                                    1093.303957,  956.377917,  924.914727,  935.285733,  977.945764,
                                    1087.225456, 1087.225456, 1242.885131]),
                           y=array([  11.927261,   16.697983,   17.045523,   19.961758,   17.762454,
                                      20.801346,   22.743581,   25.735747,   25.913071,   33.408490,
                                      36.029649,   32.280951,   27.376362,   33.638681,   37.545049,
                                      37.545049,   48.073810,   40.213674,   44.576438,   51.490797,
                                      53.656490,   32.503372,   41.050653,   43.968449,   50.788502,
                                      48.405048,   38.590562,   36.527862,   43.968449,   47.093636,
                                      53.289317,   57.077012,   60.715587,   60.300108,   63.705209,
                                      50.788502,   57.866264,   64.586115,   60.715587,   84.419223,
                                      81.570066,   81.570066,   76.156984]),
                           label='Galloway et al 1957')

# De Acetis & Thodos 1960 (as reported by Wakao et al 1979):
expt_Nu['De Acetis'] = dict(x=array([  32.426187,   90.957344,   90.957344,  157.960032,
                                      291.669516,  563.125672,  765.204014, 2170.509761]),
                            y=array([  22.743581,   22.900288,   26.452407,   31.191465,
                                       40.769742,   55.150657,   67.766322,  103.730046]),
                            label='De Acetis & Thodos 1960')

# McConnachie & Thodos 1963 (as reported by Wakao et al 1979):
expt_Nu['McConnachie'] = dict(x=array([ 146.099560,  294.939986,  400.779564,  967.101718,
                                       1519.140922]),
                              y=array([  19.961758,   37.803742,   41.618295,   70.616560,
                                         89.186313]),
                              label='McConnachie & Thodos 1963')

# Sen Gupta & Thodos 1963/1964 (as reported by Wakao et al 1979):
expt_Nu['Sen Gupta'] = dict(x=array([6191.096230, 5726.236084, 5150.678076, 5150.678076,
                                     4737.451688, 4333.151505, 2041.396476, 1919.963525,
                                     1775.802541, 1642.465923, 1405.075812, 1397.263937,
                                     1270.914172, 1081.180751,  841.277874]),
                            y=array([ 228.479331,  249.812585,  243.044548,  241.381386,
                                      214.786980,  167.745966,  118.185613,  114.196837,
                                      109.587603,  117.376864,  114.196837,  105.889011,
                                      108.092911,   98.185580,   80.457512]),
                            label='Sen Gupta & Thodos 1963/1964')

# Malling & Thados 1967 (as reported by Wakao et al 1979):
expt_Nu['Malling'] = dict(x=array([8698.959569, 5507.063660, 4711.112624, 4052.734785, 2551.406520,
                                   1136.815725,  559.994832,  288.435310,  187.762019]),
                          y=array([ 286.586983,  203.306420,  184.672228,  179.669004,  131.910185,
                                     89.800823,   61.979279,   46.451314,   36.779545]),
                          label='Malling & Thados 1967')

if __name__ == '__main__':
    from numpy import empty, zeros, ones, logspace, log10
    import matplotlib.pyplot as plt

    from reactor_modeling import rxn_rate_Klugherz, turb_Nu_text, turb_Nu_finlayson, turb_Nu_Kramers
    from parameter_set import ParameterSet, get_default_params, set_default_params

    # Reaction-rate law — plot models & expts.
    Pe = expt_rxn['Pres. C2H4 [atm]'].to_numpy()
    Po = expt_rxn['Pres. O2 [atm]'].to_numpy()
    Pp = expt_rxn['Pres. Prod. [atm]'].to_numpy()
    X = array([Pe, Po, zeros(Pe.shape), Pp / 2, Pp / 2,
               expt_rxn_aux['Pinert'] * ones(Pe.shape)]).T  # attribute the inert to nitrogen
    P = X.sum(axis=1) * 1.01325e5
    X /= X.sum(axis=1).reshape((-1, 1))
    T = expt_rxn_aux['Tavg'] + 273.15  # reactor temperature [K]

    y_e = expt_rxn[['rel. rate C2H4O [-]', 'rel. rate CO2 [-]']].to_numpy()  # normalized rates

    # Reference conditions & rate [mol rxn. / (s m^2-cat.)]
    x_ref = array([expt_rxn_aux['Peth_ref'], expt_rxn_aux['Poxy_ref'],
                   0, 0, 0, expt_rxn_aux['Pinert']])
    P_ref = x_ref.sum() * 1.01325e5
    x_ref /= x_ref.sum()
    T_ref = expt_rxn_aux['Tavg'] + 273.15  # reactor temperature [K]
    R_ref = array([expt_rxn_aux['Rethox_ref'], expt_rxn_aux['Rco2_ref']]) / (60 * 0.5)

    PSS = type('ParameterSetSubclass1', (ParameterSet,), dict())
    θ = PSS(**get_default_params((rxn_rate_Klugherz)))

    R_m_ref = array([((6 + 7) / 2 * 1e-6), ((4 + 6) / 2 * 1e-6)]) / (60 * 0.5)

    y_m = empty(y_e.shape)
    for i in range(y_m.shape[0]):
        y_m[i] = rxn_rate_Klugherz(X[i], P[i], T, **set_default_params(rxn_rate_Klugherz, θ))
    y_m /= R_m_ref

    plt.figure(figsize=(5, 6))
    lookup = {0.061:'^', 0.132:'s', 0.263:'o', 0.526:'^', 0.789:'s', 1.05:'o', 1.32:'v'}
    for i in range(y_m.shape[0]):
        marker = lookup[Po[i]]
        plt.plot([Pe[i], Pe[i]], [y_e[i, 0], y_m[i, 0]], linewidth=0.5, color='black')
        plt.plot(Pe[i], y_e[i, 0], marker=marker, markerfacecolor='none',
                 color='black', linestyle=None)
        plt.plot(Pe[i], y_m[i, 0], marker=marker, markerfacecolor=None,
                 color='black', linestyle=None)
    plt.xlim(0, 1.4)
    plt.ylim(0, 3.2)


    # Turbulent Nu — plot models & expts.
    plt.figure(figsize=(9, 6))
    for auth, expt in expt_Nu.items():
        plt.loglog(expt['x'], expt['y'], 'o', markersize=3, label=expt['label'])
    Pr0 = 0.71
    ε0 = 0.35
    Re_t = logspace(log10(100), log10(1e4), 200)
    Nu_t = turb_Nu_text(Re_t, Pr0, ε0)
    plt.loglog(Re_t, Nu_t, label='Textbook Model')
    Re_f = logspace(log10(0.01), log10(50), 200)
    Nu_f = turb_Nu_finlayson(Re_f, Pr0, ε0)
    Nu_fcorr = turb_Nu_finlayson(Re_f / ε0, Pr0, ε0)
    plt.loglog(Re_f, Nu_f, linestyle='-', color='tab:orange', label='Finlayson, 1987')
    plt.loglog(Re_f, 1.6 * Nu_f, linestyle='--', color='tab:orange', label='Finlayson, fa applied')
    plt.loglog(Re_f, 1.6 * Nu_fcorr, linestyle='-.', color='tab:orange', label='Finlayson, superficial')
    Re_k = logspace(log10(100), log10(3500), 200)
    Nu_k = turb_Nu_Kramers(Re_k, Pr0, ε0)
    Nu_kcorr = turb_Nu_Kramers(Re_k / ε0, Pr0, ε0)
    plt.loglog(Re_k, Nu_k, linestyle='-', color='tab:green', label='Thoenes & Kramers, 1958')
    plt.loglog(Re_k, 1.6 * Nu_k, linestyle='--', color='tab:green', label='Thoenes & Kramers, fa applied')
    plt.loglog(Re_k, 1.6 * Nu_kcorr, linestyle='-.', color='tab:green', label='Thoenes & Kramers, superficial')
    plt.xlabel('Reynolds Number, Re [-]', fontsize=16)
    plt.ylabel('Nusselt Number, Nu [-]', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.25)
    
    plt.show()