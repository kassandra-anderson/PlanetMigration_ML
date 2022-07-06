import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as imbl_pipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import matthews_corrcoef, make_scorer
import train_tune_score as tts
from scipy.stats import uniform, loguniform, randint
import pickle

seed = 1
testsize = "0.33"

# Set number of splits for cross validation
n_splits = 5

# search_method = "grid"
search_method = "random"

# Specify scaling, resample method, cross validation properties, and score metric
scaler_name, scaler = "standardscaler", StandardScaler()
# scaler_name, scaler = "minmaxscaler", MinMaxScaler()

resampler_name, resampler = "smote", SMOTE(random_state=seed)
# resampler_name, resampler = "smoteenn", SMOTEENN(random_state=seed)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
score = make_scorer(matthews_corrcoef)

# Directory to store trained/tuned models
outpath = "/Users/kassand/astro/PlanetMigration_ML/tuned_models/feature_exploration/"

path_to_data = "/Users/kassand/astro/PlanetMigration_ML/training_data/"
df_train = pd.read_pickle(
    path_to_data + "train_data_seed_" + str(seed) + "_testsize_" + testsize + ".pkl"
)
ylabel = "flag"

# Group features by different categories

# These are the raw input variables (mostly orbital elements)
orb_features = [
    "m2_input",
    "m3_input",
    "a1_input",
    "a2_input",
    "e2_input",
    "Pstar1_input",
    "I1_input",
    "I2_input",
    "node1_input",
    "node2_input",
    "peri1_input",
    "peri2_input",
    "cos_Imut_input",
]

# Strengths of SRFs and octupole strength
eps_features = [
    "log_eps_gr_input",
    "log_eps_tide1_input",
    "log_eps_tide2_input",
    "log_eps_rot1_input",
    "log_eps_rot2_input",
    "log_eps_oct_input",
]

# Mutual inclination
cosI_features = ["cos_Imut_input"]

# Features obtrained from calculating the maximum eccentricity
emax_features = [
    "log_peri_max_over_rtide_input",
    "log_aF_max_input",
    "log_aF_max7_input",
]

# Powers of aF = a(1 - emax^2)
emax_powers_features = [
    "log_peri_max_over_rtide_input",
    "log_aF_max_input",
    "log_aF_max2_input",
    "log_aF_max3_input",
    "log_aF_max4_input",
    "log_aF_max5_input",
    "log_aF_max6_input",
    "log_aF_max7_input",
]

# Try all these combinations of features
feature_dict = {
    "inputvars": orb_features,
    "eps+inc": eps_features + cosI_features,
    "inputvars+eps": orb_features + eps_features,
    "inputvars+eps+emax": orb_features + eps_features + emax_features,
    "eps+emax": eps_features + emax_features,
    "eps+emax+inc": eps_features + cosI_features + emax_features,
    "eps+emax-powers": eps_features + emax_powers_features,
    "eps+emax-powers+inc": eps_features + emax_powers_features + cosI_features,
}

# delete later
# feature_dict = {"eps_emax_rtide":eps_features + emax_features + ["Rtide_input"]}

# Print the numbers of each outcome
print("Class counts in training set: \n", df_train[ylabel].value_counts())

y_train = df_train[ylabel]

# Specify ML models and hyperparameter grids
RF = RandomForestClassifier(random_state=seed)
params_RF = {
    "RF__n_estimators": randint(10, 100),
    "RF__min_samples_leaf": randint(2, 15),
    "RF__criterion": ["gini", "entropy"],
}
name, model_info = "RF", (RF, params_RF, 50)

# LR = LogisticRegression(penalty='l2', solver='liblinear', random_state=seed)
# params_LR = {"LR__C" : loguniform(0.01,100),
#             "LR__penalty" : ['l2', 'l1']}
# name, model_info = "LR", (LR, params_LR, 20)

model_cvs = {}
for (
    feature_name,
    features,
) in feature_dict.items():

    print("Trying " + name, "using features " + feature_name)
    ml_model, model_params, n_iter = model_info
    print("ml_model = ", ml_model)
    print("scaler: ", scaler)
    print("resampler: ", resampler)

    # Set features to use in model training
    X_train = df_train[features]

    # Create imblearn pipeline
    pipeline = imbl_pipeline(
        [("scaler", scaler), ("resampler", resampler), (name, ml_model)]
    )
    print("Made pipeline")

    # Fit model and tune hyperparameters
    print("Tuning model...")
    model_cvs[name] = tts.train_model(
        X_train, y_train, pipeline, model_params, cv, score, seed, n_iter, search_method
    )

    # Pickle the pipeline
    outfile = (
        outpath
        + name
        + "_"
        + scaler_name
        + "_"
        + resampler_name
        + "_features_"
        + feature_name
        + "_seed_"
        + str(seed)
        + "_testsize_"
        + testsize
        + "_cv_nsplits_"
        + str(n_splits)
        + ".pkl"
    )
    with open(outfile, "wb") as file:
        pickle.dump(model_cvs[name], file=file)

    print("Finished tuning model \n ----------------------------------------------")
