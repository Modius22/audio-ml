from sklearn import ensemble, linear_model, model_selection, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from libs import utils


# Step 1. Define an objective function to be maximized.
def objective(trial, models, smote):
    X, y = utils.get_trainings_data(smote=smote)

    # classifier_name = trial.suggest_categorical("classifier", ["LogReg","DeTree", "RandomForest"])
    classifier_name = trial.suggest_categorical("classifier", models)

    # Step 2. Setup values for the hyperparameters:
    if classifier_name == "LogReg":
        logreg_c = trial.suggest_float("logreg_c", 1e-10, 1e10, log=True)
        logreg_solver = trial.suggest_categorical("logreg_solver", ["newton-cg", "lbfgs", "liblinear"])
        logreg_penalty = trial.suggest_categorical("logreg_penalty", ["l2"])
        classifier_obj = linear_model.LogisticRegression(C=logreg_c, solver=logreg_solver, penalty=logreg_penalty)
    elif classifier_name == "DeTree":
        # dt_criterion = trial.suggest_categorical('gini', 'entropy')
        dt_max_depth = trial.suggest_int("dt_max_depth", 1, 20)
        dt_min_samples_leaf = trial.suggest_int("dt_min_samples_leaf", 1, 5)
        dt_min_weight_fraction_leaf = trial.suggest_float("dt_min_weight_fraction_leaf", 0.0, 0.5)
        classifier_obj = tree.DecisionTreeClassifier(
            # criterion=dt_criterion,
            max_depth=dt_max_depth,
            min_samples_leaf=dt_min_samples_leaf,
            min_weight_fraction_leaf=dt_min_weight_fraction_leaf,
        )
    elif classifier_name == "SVC":
        svc_C = trial.suggest_float("svc_C", 0.1, 20)
        svc_gamma = trial.suggest_float("svc_gamma", 0.00001, 10)
        svc_kernel = trial.suggest_categorical("svc_kernel", ["rbf", "sigmoid"])
        classifier_obj = SVC(C=svc_C, gamma=svc_gamma, kernel=svc_kernel)
    elif classifier_name == "GaussianNB":
        gnb_var_smoothing = trial.suggest_float("gnb_var_smoothing", 1e-10, 1e10, log=True)
        classifier_obj = GaussianNB(var_smoothing=gnb_var_smoothing)
    elif classifier_name == "MLP":
        mlp_activation = trial.suggest_categorical("mlp_activation", ["tanh", "relu"])
        mlp_solver = trial.suggest_categorical(
            "solver",
            ["sgd", "adam"],
        )
        mlp_alpha = trial.suggest_float("alpha", 0.0001, 0.5)
        mlp_learning_rate = trial.suggest_categorical("mlp_learning_rate", ["constant", "adaptive"])

        n_layers = trial.suggest_int("n_layers", 1, 4)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_{i}", 1, 100))

        classifier_obj = MLPClassifier(
            hidden_layer_sizes=tuple(layers),
            activation=mlp_activation,
            solver=mlp_solver,
            alpha=mlp_alpha,
            learning_rate=mlp_learning_rate,
        )

    elif classifier_name == "XGBClassifier":
        XGBClassifier_n_estimators = trial.suggest_int("XGBClassifier_n_estimators", 1, 1000)
        XGBClassifier_max_depth = trial.suggest_int("XGBClassifier_max_depth", 1, 200)
        XGBClassifier_learning_rate = trial.suggest_float("XGBClassifier_learning_rate", 0.01, 20)
        classifier_obj = XGBClassifier(
            n_estimators=XGBClassifier_n_estimators,
            max_depth=XGBClassifier_max_depth,
            learning_rate=XGBClassifier_learning_rate,
        )
    else:
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000)
        rf_max_depth = trial.suggest_int("rf_max_depth", 10, 100)
        rf_max_features = trial.suggest_int("rf_max_features", 1, 10)
        rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)
        rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 12)
        rf_random_state = trial.suggest_int("random_state", 0, 42)
        rf_weights = trial.suggest_int("rf_weights", 0, 400)
        classifier_obj = ensemble.RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            max_features=rf_max_features,
            min_samples_leaf=rf_min_samples_leaf,
            min_samples_split=rf_min_samples_split,
            random_state=rf_random_state,
            class_weight={0: rf_weights, 1: 1},
        )

    # Step 3: Scoring method:
    score = model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=10, scoring="f1")
    f1 = score.mean()
    return f1
