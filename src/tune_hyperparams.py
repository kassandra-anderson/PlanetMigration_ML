import numpy as np
import pandas as pd

# import process_features
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

# Set search_method to either "random" or "grid"
search_method = "random"

# Specify scaling, resample method, cross validation properties, and score metric
# scaler_name, scaler = "standardscaler", StandardScaler()
scaler_name, scaler = "minmaxscaler", MinMaxScaler()

resampler_name, resampler = "smote", SMOTE(random_state=seed)
# resampler_name, resampler = "smoteenn", SMOTEENN(random_state=seed)

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
score = make_scorer(matthews_corrcoef)

# Directory to store trained/tuned models
outpath = "../tuned_models/model_exploration/"

path_to_data = "../training_data/"
df_train = pd.read_pickle(
    path_to_data + "train_data_seed_" + str(seed) + "_testsize_" + testsize + ".pkl"
)
ylabel = "flag"

feature_names = [
    "log_eps_gr_input",
    "log_eps_tide1_input",
    "log_eps_tide2_input",
    "log_eps_rot1_input",
    "log_eps_rot2_input",
    "log_eps_oct_input",
    "cos_Imut_input",
    "log_emax_input",
    "log_peri_max_over_rtide_input",
    "log_aF_max_input",
    "log_aF_max7_input",
]

# Print the numbers of each outcome
print("Class counts in training set: \n", df_train[ylabel].value_counts())

# Set features to use in model training
X_train = df_train[feature_names]
y_train = df_train[ylabel]

# Specify ML models and hyperparameter grids
LR = LogisticRegression(penalty="l2", solver="liblinear", random_state=seed)
KNN = KNeighborsClassifier()
RF = RandomForestClassifier(random_state=seed)
SVC = SVC(random_state=seed, probability=True, max_iter=1000)

# if search_method is "random"
params_LR = {"LR__C": loguniform(0.01, 100), "LR__penalty": ["l2", "l1"]}
params_KNN = {"KNN__n_neighbors": randint(3, 10), "KNN__p": [1, 2]}
params_RF = {
    "RF__n_estimators": randint(10, 100),
    "RF__min_samples_leaf": randint(2, 15),
    "RF__criterion": ["gini", "entropy"],
}
params_SVC = {"SVC__C": loguniform(0.01, 100), "SVC__gamma": loguniform(0.01, 100)}


# Create a dictionary of different models to loop through and train/tune
models_dict = {
    "LR": (LR, params_LR, 20),
    "KNN": (KNN, params_KNN, 20),
    "SVC": (SVC, params_SVC, 50),
    "RF": (RF, params_RF, 50),
}


model_cvs = {}
for name, model in models_dict.items():
    print("Trying " + name)
    ml_model, model_params, n_iter = model
    print("ml_model = ", ml_model)
    print("scaler: ", scaler)
    print("resampler: ", resampler)

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
