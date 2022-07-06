def resample_data(X_train, y_train, feature_names, class_label, resample_method=None, seed=1):
    """ Resamples training data using upsampling or downsampling. X_train, y_train must be numpy arrays. """

    from sklearn.utils import resample, shuffle
    import pandas as pd

    # combine them back into a dataframe for resampling
    df_train = pd.DataFrame(X_train, columns=feature_names)

    print("Shape of X_train, shape of df_train features, length of y_train: ", X_train.shape, df_train.shape,
          len(y_train))

    df_train.insert(len(df_train.columns), class_label, y_train)

    # Separate minority and majority classes
    df_train_neg = df_train[df_train[class_label] == 0]
    df_train_pos = df_train[df_train[class_label] == 1]
    print("Total number of training examples: ", len(df_train))
    print("Number of training examples that are pos: ", len(df_train_pos))
    print("Number of training examples that are neg: ", len(df_train_neg))

    if resample_method == "up":
        # upsample minority class
        df_train_pos_up = resample(df_train_pos,
                                   replace=True,  # sample with replacement
                                   n_samples=len(df_train_neg),  # match number in majority class
                                   random_state=seed)  # reproducible results

        # combine majority and upsampled minority
        df_train_resamp = pd.concat([df_train_neg, df_train_pos_up])

    elif resample_method == "down":
        df_train_neg_down = resample(df_train_neg,
                                     replace=False,
                                     n_samples=len(df_train_pos),  # match number in minority class
                                     random_state=seed)  # reproducible results

        # Combine majority and upsampled minority
        df_train_resamp = pd.concat([df_train_neg_down, df_train_pos])

    elif resample_method == None:
        df_train_resamp = pd.concat([df_train_neg, df_train_pos])

    else:
        print('Error: Invalid resample_method.')
        return

    df_train_resamp = shuffle(df_train_resamp, random_state = seed)

    # Print new class counts
    print("New counts after resampling: \n", df_train_resamp[class_label].value_counts())

    return df_train_resamp


def plot_learning_curve(X, y, ml_model, cv=3, scoring='neg_log_loss', random_state=1):
    """ Calculates and plots a learning curve with 1-sigma uncertainties """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores, = learning_curve(ml_model,
                                                             X,
                                                             y,
                                                             train_sizes=np.linspace(0.5, 1, 10),
                                                             shuffle=True,
                                                             cv=cv,
                                                             scoring=scoring,
                                                             random_state=random_state)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning curve")
    plt.plot(train_sizes, train_scores_mean, 'b', label='train')
    plt.plot(train_sizes, test_scores_mean, 'r', label='test')

    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="b",
                     )

    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color="r",
                     )

    plt.legend()
    plt.xlabel('Training size')
    plt.ylabel('score')

    return


def plot_vary_hyper_param(X_train, y_train, X_test, y_test, ml_model, param_list, param_name):


    """Train model and calculate metrics for a range of parameter values. Plot results. """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score, log_loss

    train_acc = np.empty(len(param_list))
    test_acc = np.empty(len(param_list))
    f1_train = np.empty(len(param_list))
    f1_test = np.empty(len(param_list))
    loss_train = np.empty(len(param_list))
    loss_test = np.empty(len(param_list))

    for i, p in enumerate(param_list):

        ml_model.set_params(**{param_name:p})
        ml_model.fit(X_train, y_train)
        train_acc[i] = ml_model.score(X_train, y_train)
        test_acc[i] = ml_model.score(X_test, y_test)
        y_pred_test = ml_model.predict(X_test)
        y_pred_train = ml_model.predict(X_train)
        f1_test[i] = f1_score(y_test, y_pred_test)
        f1_train[i] = f1_score(y_train, y_pred_train)
        loss_test[i] = log_loss(y_test, y_pred_test)
        loss_train[i] = log_loss(y_train, y_pred_train)

    plt.figure()
    plt.title('Logistic Regression: Varying regularization')
    plt.plot(param_list, loss_train, label='Training Accuracy')
    plt.plot(param_list, loss_test, label='Testing Accuracy')
    plt.gca().set_xscale('log')
    plt.legend()
    plt.xlabel(param_name)
    plt.ylabel('Loss function')
    plt.show()

    plt.figure()
    plt.title('Logistic Regression: Varying regularization')
    plt.plot(param_list, f1_train, label="train")
    plt.plot(param_list, f1_test, label="test")
    plt.gca().set_xscale('log')
    plt.legend()
    plt.xlabel(param_name)
    plt.ylabel('f1 score')
    plt.show()