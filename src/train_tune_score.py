from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def train_model(
    X_train, y_train, pipeline, parameters, cv, score, seed, n_iter=10, search="random"
):

    """Fits the training data and performs hyperparameter tuning using either a grid search or randomized search.

    Input:
    X_train, y_train: training data
    X_test, y_test: test data
    pipeline: pipeline object
    parameters: hyperparameters to be tuned using either GridSearchCV or RandomizedSearchCV
    cv: cross validation method
    score: scoring metric
    seed: random number seed
    n_iter: number of parameters to be sampled in RandomizedSearchCV (ignored if search is set to "random)"
    search: either "random" (hyperparameters tuned using RandomizedSearchCV) or "grid" ("GridSearchCV"). Default is "random"

    Returns: the pipeline with the optimal hyperparameters
    """

    # Create either a GridSearchCV or RandomizedSearchCV object
    if search == "grid":
        search_cv = GridSearchCV(pipeline, parameters, cv=cv, scoring=score)
    else:
        search_cv = RandomizedSearchCV(
            pipeline, parameters, n_iter=n_iter, cv=cv, scoring=score, random_state=seed
        )

    # Fit to the training set and get predictions for the test set
    search_cv.fit(X_train, y_train)

    print("Best parameters: {}".format(search_cv.best_params_))
    print("Best score is {}".format(search_cv.best_score_))

    # Update the pipeline with the best parameters and fit
    pipeline.set_params(**search_cv.best_params_)
    pipeline.fit(X_train, y_train)

    print("Printing updated pipeline params: \n", pipeline.get_params())

    return pipeline
