import numpy as np
import pandas as pd

def calculate_angles(df):

    L1x, L1y, L1z = df["L1x_input"], df["L1y_input"], df["L1z_input"]
    L2x, L2y, L2z = df["L2x_input"], df["L2y_input"], df["L2z_input"]

    e1x, e1y, e1z = df["e1x_input"], df["e1y_input"], df["e1z_input"]
    e2x, e2y, e2z = df["e2x_input"], df["e2y_input"], df["e2z_input"]

    L1 = np.sqrt(L1x ** 2 + L1y ** 2 + L1z ** 2)
    L2 = np.sqrt(L2x ** 2 + L2y ** 2 + L2z ** 2)

    e1 = np.sqrt(e1x ** 2 + e1y ** 2 + e1z ** 2)
    e2 = np.sqrt(e2x ** 2 + e2y ** 2 + e2z ** 2)

    l1_hatx, l1_haty, l1_hatz = L1x / L1, L1y / L1, L1z / L1
    l2_hatx, l2_haty, l2_hatz = L2x / L2, L2y / L2, L2z / L2

    e1_hatx, e1_haty, e1_hatz = e1x / e1, e1y / e1, e1z / e1
    e2_hatx, e2_haty, e2_hatz = e2x / e2, e2y / e2, e2z / e2

    cos_Imut = l1_hatx*l2_hatx + l1_haty*l2_haty + l1_hatz*l2_hatz

    Omega1 = np.arctan2(-l1_hatx, l1_haty)
    Omega2 = np.arctan2(-l2_hatx, l2_haty)

    sin_Omega1, cos_Omega1 = np.sin(Omega1), np.cos(Omega1)
    sin_Omega2, cos_Omega2 = np.sin(Omega2), np.cos(Omega2)

    sin_I1, cos_I1 = l1_hatx / sin_Omega1, l1_hatz
    sin_I2, cos_I2 = l2_hatx / sin_Omega2, l2_hatz

    sin_omega1 = e1_hatz / sin_I1
    sin_omega2 = e2_hatz / sin_I2

    cos_omega1 = (e1_hatx + sin_omega1 * cos_I1 * sin_Omega1) / cos_Omega1
    cos_omega2 = (e2_hatx + sin_omega2 * cos_I2 * sin_Omega2) / cos_Omega2

    df["cos_Imut_input"] = cos_Imut

    df["cos_I1_input"], df["sin_I1_input"] = cos_I1, sin_I1
    df["cos_I2_input"], df["sin_I2_input"] = cos_I2, sin_I2

    df["cos_Omega1_input"], df["sin_Omega1_input"] = cos_Omega1, sin_Omega1
    df["cos_Omega2_input"], df["sin_Omega2_input"] = cos_Omega2, sin_Omega2

    df["cos_omega1_input"], df["sin_omega1_input"] = cos_omega1, sin_omega1
    df["cos_omega2_input"], df["sin_omega2_input"] = cos_omega2, sin_omega2

    return df

def add_log(df, features):
    ''' Adds the log10 of each entry in list of features to dataframe df '''

    for f in features:
        df["log_" + f] = np.log10(df[f].values)

    return df


def calc_emax(df):

    '''Calculates the maximum quadrupole-order eccentricity from energy conservation'''

    from scipy.optimize import fsolve, brentq

    def energy(x, *args):

        '''x = j = sqrt(1 - e^2)'''

        eps_gr, eps_tide1, eps_tide2, cosI0, eta = args

        j = x
        jsq = j ** 2
        esq = 1. - jsq
        f = (1 + 3 * esq + (3. / 8.) * (esq ** 2)) / (j ** 9)

        Phi_gr0 = - eps_gr
        Phi_tide10 = - eps_tide1 / 15
        Phi_tide20 = - eps_tide2 / 15

        Phi_gr = - eps_gr / j
        Phi_tide1 = - eps_tide1 * f / (15 * j ** 9)
        Phi_tide2 = - eps_tide2 * f / (15 * j ** 9)

        Phi_srf0 = Phi_gr0 + Phi_tide10 + Phi_tide20
        Phi_srf = Phi_gr + Phi_tide1 + Phi_tide2

        prefac = (3. / 8.) * (jsq - 1) / jsq

        eq = prefac * (5 * (cosI0 + eta / 2) ** 2 - (
                3 + 4 * eta * cosI0 + (9. / 4.) * eta * eta) * jsq + eta * eta * jsq * jsq) + Phi_srf - Phi_srf0
        return eq

    L1x, L1y, L1z = df.L1x_input, df.L1y_input, df.L1z_input
    L2x, L2y, L2z = df.L2x_input, df.L2y_input, df.L2z_input
    L1 = np.sqrt(L1x ** 2 + L1y ** 2 + L1z ** 2)
    L2 = np.sqrt(L2x ** 2 + L2y ** 2 + L2z ** 2)

    eta = L1/L2 #assumes e1 = 0

    l1_hatx, l1_haty, l1_hatz = L1x / L1, L1y / L1, L1z / L1
    l2_hatx, l2_haty, l2_hatz = L2x / L2, L2y / L2, L2z / L2

    cos_Imut = l1_hatx * l2_hatx + l1_haty * l2_haty + l1_hatz * l2_hatz

    eps_gr = df.eps_gr_input
    eps_tide1 = df.eps_tide1_input
    eps_tide2 = df.eps_tide2_input

    args = (eps_gr, eps_tide1, eps_tide2, cos_Imut, eta)

    small = 1e-3
    xmin = small
    xmax = 1. - small

    try:
        j = brentq(energy, xmin, xmax, args=args)
    except ValueError:
        print("No sign change for outcome " + df.sflag + " with index " + str(df.simnum))
        print(eps_gr,cos_Imut)
        print("Trying fsolve instead")
        j = fsolve(energy,0.1, args = args)[0] - small
        print("j = ", j)
        pass

    return np.sqrt(1 - j**2)


