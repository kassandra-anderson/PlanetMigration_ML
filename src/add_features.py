import numpy as np
import pandas as pd
import process_features
from sklearn.utils import shuffle

# Fixed masses
path_to_data = "../training_data/"
df = pd.read_csv(path_to_data + "alldata.csv")
df = shuffle(df,random_state=0)

emax = np.zeros(len(df.index.values))
for i, dfi in enumerate(df.itertuples()):
    root = process_features.calc_emax(dfi)
    emax[i] = root

df["emax_input"] = emax
df["jmax_input"] = np.sqrt(1 - emax**2)
df["peri_max_input"] = df["a1"]*(1 - emax)
df["peri_max_over_rtide_input"] = df["a1"]*(1 - emax)/df["Rtide_input"]
df["aF_max_input"] = df["a1"]*(1 - emax**2)
df["aF_max2_input"] = df["aF_max_input"]**2
df["aF_max3_input"] = df["aF_max_input"]**3
df["aF_max4_input"] = df["aF_max_input"]**4
df["aF_max5_input"] = df["aF_max_input"]**5
df["aF_max6_input"] = df["aF_max_input"]**6
df["aF_max7_input"] = df["aF_max_input"]**7
df["aF_max8_input"] = df["aF_max_input"]**8
df["aF_max9_input"] = df["aF_max_input"]**9

# Get sines and cosines of all the angles
df = process_features.calculate_angles(df)

# Add log scaled features
df = process_features.add_log(df,
                              ["emax_input",
                               "jmax_input",
                               "peri_max_input",
                               "peri_max_over_rtide_input",
                               "aF_max_input",
                               "aF_max2_input",
                               "aF_max3_input",
                               "aF_max4_input",
                               "aF_max5_input",
                               "aF_max6_input",
                               "aF_max7_input",
                               "eps_gr_input",
                               "eps_tide1_input",
                               "eps_tide2_input",
                               "eps_rot1_input",
                               "eps_rot2_input",
                               "eps_oct_input"])

emax = df["emax_input"]
jmax = np.sqrt(1 - emax**2)
aperi = df["a1"]*(1 - emax)
aconst = df["a1"]*(1 - emax**4)


outfile = path_to_data + "alldata_additional_features.csv"
df.to_csv(outfile)
