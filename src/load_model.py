import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import matthews_corrcoef

############
path = "/Users/kassand/astro/PlanetMigration_ML/tuned_models/"

ylabel = "flag"





# for plotting decision boundary
#feature1 = "cos_Imut_input"
#feature2 = "log_eps_oct_input"
#x1 = df_test[feature1]#.values
#x2 = df_test[feature2]#.values
#xx, yy = np.meshgrid(np.linspace(np.min(x1), np.max(x1), 100), np.linspace(np.min(x2), np.max(x2), 100))

###########

names = ["LR", "KNN", "RF"]
scaler = "standardscaler"
resampler = "smote"

fnames = [path + n + "_" + scaler + "_" + resampler + ".pkl" for n in names]
print("fnames = \n",fnames)
#fnames = [path + "LR_standardscaler_smote.pkl",
#          path + "KNN_standardscaler_smote.pkl",
#          path + "SVC_standardscaler_smote.pkl",
#          path + "RF_standardscaler_smote.pkl"]

#fnames = [path + "LR_standardscaler_smote.pkl"]
#fnames = [path + "LR_minmaxscaler_smote.pkl"]
#fnames = [path + "RF_standardscaler_smote.pkl"]

model_info = {"name":names, "fname":fnames}
print("model_info = \n",model_info.items())
# Loop through trained models
#for fname in fnames:
#for name, fname in model_info.items():
for name in names:
    fname = path + name + "_" + scaler + "_" + resampler + ".pkl"
    print("fname = ", fname)
    with open(fname, 'rb') as file:
        model = pickle.load(file)
    print("model loaded ", model)


    #print("coefs: ",coefs.shape)


    #print(model.get_params()["LR"].random_state)

    feature_names = list(model.feature_names_in_)
    fn_short = [s.replace("_input", "") for s in model.feature_names_in_]

    # Load pickled training and test data
    df_train = pd.read_pickle(path + "train_data.pkl")
    df_test = pd.read_pickle(path + "test_data.pkl")

    X_train, y_train = df_train[feature_names], df_train[ylabel]
    X_test, y_test = df_test[feature_names], df_test[ylabel]

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mscore_train = matthews_corrcoef(y_train, y_train_pred)
    mscore_test = matthews_corrcoef(y_test, y_test_pred)

    print(mscore_train, mscore_test)


    #Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z = Z.reshape(xx.shape)
    #out = plt.contourf(xx, yy, Z, **params)
    #Z = model.decision_function(np.c_[X,y])
    #Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #Z = Z.reshape(xx.shape)
    #plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu)
    #plt.scatter(x1, x2, c=y, cmap=plt.cm.RdBu_r, edgecolors="k")

    try:
        coefs = model.named_steps[name].coef_
        plt.title(name)
        plt.plot(np.abs(coefs[0, :]),'k.')
        plt.plot(np.abs(coefs[1, :]), 'r.')
        plt.plot(np.abs(coefs[2, :]), 'b.')
        plt.axhline(0,color='k',linestyle='--')
        plt.gca().set_xticks(np.arange(0,11))
        plt.gca().set_xticklabels(fn_short, rotation=45, fontsize=10)
        plt.subplots_adjust(bottom=0.2)

    except AttributeError:
        print("no coefs")

    print("----------------------------------------------")
plt.show()