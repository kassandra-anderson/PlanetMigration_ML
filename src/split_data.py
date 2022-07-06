def split_store_data(
    fname,
    ylabel,
    outpath="../training_data/",
    test_size=0.33,
    seed=0,
    return_df=False,
    save_pkl=False,
):

    """Takes the input data and splits into training and test sets according to the desired test_size.
    Returns dataframes containing the training and test sets if return_df is set to True.
    Pickles these dataframes if save_pkl is set to True."""

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    df = pd.read_csv(fname)
    df = shuffle(df, random_state=seed)

    # Remove planets in the process of migrating, since they are a negligibly small fracton of all cases
    df = df[df["flag"] < 3]

    X = df.drop(ylabel, axis=1)
    y = df[ylabel]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Create dataframes of train and test data and pickle for use later
    df_train = pd.DataFrame(X_train)
    df_train.insert(len(df_train.columns), ylabel, y_train)

    df_test = pd.DataFrame(X_test)
    df_test.insert(len(df_test.columns), ylabel, y_test)

    if save_pkl:
        outstring = "_seed_" + str(seed) + "_testsize_" + str(test_size) + ".pkl"
        df_train.to_pickle(outpath + "train_data" + outstring)
        df_test.to_pickle(outpath + "test_data" + outstring)

    if return_df:
        return df_train, df_test
    else:
        return


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1])
    fname = "../training_data/alldata_additional_features.csv"
    split_store_data(fname, "flag", seed=seed, save_pkl=True)
