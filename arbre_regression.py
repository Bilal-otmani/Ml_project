from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "hour.csv"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
pd.set_option("display.max_columns", None)

def load_prepared_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    data = data.sort_values(["dteday", "hr"]).reset_index(drop=True)
    data["datetime"] = data["dteday"] + pd.to_timedelta(data["hr"], unit="h")

    # Reconstitue les heures manquantes pour que les lags soient calendaires.
    hourly_count = data.set_index("datetime")["cnt"].asfreq("h")
    for lag in [1, 24, 168]:
        data[f"lag_{lag}"] = data["datetime"].map(hourly_count.shift(lag))

    data["lag_mean_24_168"] = data[["lag_24", "lag_168"]].mean(axis=1)
    data["lag_delta_week"] = data["lag_24"] - data["lag_168"]
    data = data.dropna(subset=["lag_1", "lag_24", "lag_168"]).reset_index(drop=True)

    split_index = int(0.8 * len(data))
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    excluded = ["cnt", "casual", "registered", "instant", "dteday", "datetime"]
    X_train = train.drop(columns=excluded)
    y_train = train["cnt"]
    X_test = test.drop(columns=excluded)
    y_test = test["cnt"]

    return data, train, test, X_train, y_train, X_test, y_test


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def add_time_groups(frame, hour_column="hr"):
    frame = frame.copy()
    frame["tranche"] = pd.cut(
        frame[hour_column],
        bins=[-1, 5, 9, 15, 19, 23],
        labels=["Nuit 0-5h", "Matin 6-9h", "Journee 10-15h",
                "Soir 16-19h", "Soiree 20-23h"],
        ordered=True,
    )
    return frame


def main():
    data, train, test, X_train, y_train, X_test, y_test = load_prepared_data()
    print("Data \n ", data.head())
    print("Train \n",train.head())
    print("Train X \n",X_train.head())
    print("Test  X \n",X_test.head())
    print(f"Lignes totales: {len(data)}")
    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"Periode train: {train['dteday'].min().date()} -> {train['dteday'].max().date()}")
    print(f"Periode test: {test['dteday'].min().date()} -> {test['dteday'].max().date()}")
    print(f"Variables d'entree: {list(X_train.columns)}")

    # Point de comparaison: arbre illimite, sans prise en compte du jeu test.
    baseline = DecisionTreeRegressor(random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_train_pred = baseline.predict(X_train)
    baseline_train_metrics = regression_metrics(y_train, baseline_train_pred)
    print("\nArbre de base - apprentissage:")
    print(json.dumps(baseline_train_metrics, indent=2))
    print(f"Profondeur du modele de base: {baseline.get_depth()}")
    print(f"Feuilles du modele de base: {baseline.get_n_leaves()}")

    # Reglage sur TimeSeriesSplit, exclusivement sur le train.
    parameter_grid = {
        "max_depth": [6, 8, 10, 12, 15, 20],
        "min_samples_leaf": [2, 3, 5, 8, 10, 15],
        "min_samples_split": [2, 10, 20],
        "criterion": ["squared_error"],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        refit=True,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    best_depth = grid_search.best_params_["max_depth"]
    best_leaf = grid_search.best_params_["min_samples_leaf"]
    best_split = grid_search.best_params_["min_samples_split"]
    best_criterion = grid_search.best_params_["criterion"]
    cv_rmse = -grid_search.cv_results_["mean_test_score"]
    print("\nMeilleure profondeur:", best_depth)
    print("Meilleur min_samples_leaf:", best_leaf)
    print("Meilleur min_samples_split:", best_split)
    print("Meilleur critere:", best_criterion)
    print(f"RMSE CV moyenne: {cv_rmse.min():.4f}")

    cv_table = pd.DataFrame({
        "max_depth": [
            "Illimitee" if value is None else str(value)
            for value in grid_search.cv_results_["param_max_depth"]
        ],
        "min_samples_leaf": grid_search.cv_results_["param_min_samples_leaf"],
        "min_samples_split": grid_search.cv_results_["param_min_samples_split"],
        "criterion": grid_search.cv_results_["param_criterion"],
        "RMSE_CV": -grid_search.cv_results_["mean_test_score"],
        "RMSE_CV_std": grid_search.cv_results_["std_test_score"],
    })
    cv_table.to_csv(BASE_DIR / "cv_max_depth.csv", index=False)

    # Une seule prediction sur le test chronologique, avec le meilleur modele reajuste.
    final_model = grid_search.best_estimator_
    train_predictions = final_model.predict(X_train)
    train_metrics = regression_metrics(y_train, train_predictions)
    test_predictions = final_model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_predictions)
    print("\nArbre optimise - apprentissage:")
    print(json.dumps(train_metrics, indent=2))
    print("\nArbre optimise - test:")
    print(json.dumps(test_metrics, indent=2))

    pd.DataFrame([{
        "Modele": "Arbre de regression",
        "max_depth": best_depth if best_depth is not None else "Illimitee",
        "min_samples_leaf": best_leaf,
        "min_samples_split": best_split,
        "criterion": best_criterion,
        **test_metrics,
    }]).to_csv(BASE_DIR / "resultats_arbre.csv", index=False)

    prediction_output = test[["dteday", "hr", "cnt"]].copy()
    prediction_output = prediction_output.rename(columns={"cnt": "y_reel"})
    prediction_output["y_predit"] = test_predictions
    prediction_output["erreur_absolue"] = np.abs(
        prediction_output["y_reel"] - prediction_output["y_predit"]
    )
    prediction_output["erreur_carree"] = (
        prediction_output["y_reel"] - prediction_output["y_predit"]
    ) ** 2
    prediction_output.to_csv(BASE_DIR / "predictions_test_arbre.csv", index=False)

    with open(BASE_DIR / "resultats_arbre.json", "w", encoding="utf-8") as handle:
        json.dump({
            "max_depth": best_depth,
            "min_samples_leaf": best_leaf,
            "min_samples_split": best_split,
            "criterion": best_criterion,
            "baseline_train_metrics": baseline_train_metrics,
            "baseline_depth": baseline.get_depth(),
            "tuned_train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }, handle, indent=2, ensure_ascii=False)

    # Courbe RMSE de validation croisee.
    cv_by_depth = (
        cv_table
        .groupby("max_depth", as_index=False)["RMSE_CV"]
        .min()
    )
    numeric_order = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "8": 5,
                     "10": 6, "12": 7, "15": 8, "20": 9, "Illimitee": 10}
    cv_by_depth["_order"] = cv_by_depth["max_depth"].map(numeric_order)
    cv_by_depth = cv_by_depth.sort_values("_order")
    depth_labels = cv_by_depth["max_depth"].tolist()
    plt.figure(figsize=(10, 5))
    x_positions = np.arange(len(depth_labels))
    plt.plot(
        x_positions,
        cv_by_depth["RMSE_CV"],
        marker="o",
    )
    plt.xticks(x_positions, depth_labels)
    plt.xlabel("max_depth")
    plt.ylabel("RMSE de validation croisee")
    plt.title("Reglage de la profondeur maximale (TimeSeriesSplit)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "courbe_validation_max_depth.png", dpi=150)
    plt.close()

    # Reel vs predit sur les 14 premiers jours du test.
    first_two_weeks = prediction_output.head(24 * 14).copy()
    first_two_weeks["date_heure"] = (
        first_two_weeks["dteday"].dt.strftime("%Y-%m-%d")
        + " "
        + first_two_weeks["hr"].astype(str).str.zfill(2)
        + ":00"
    )
    first_two_weeks.index = pd.date_range(
        start=test["dteday"].iloc[0],
        periods=len(first_two_weeks),
        freq="h",
    )
    plt.figure(figsize=(14, 5))
    plt.plot(first_two_weeks.index, first_two_weeks["y_reel"],
             label="Reel", linewidth=1.5)
    plt.plot(first_two_weeks.index, first_two_weeks["y_predit"],
             label="Arbre optimise", linewidth=1.5, alpha=0.85)
    plt.xlabel("Date")
    plt.ylabel("Nombre de locations")
    plt.title("Valeurs reelles et predites - 14 premiers jours du test")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reel_vs_predit_test.png", dpi=150)
    plt.close()

    # Erreurs par heure et par tranche.
    hourly_errors = prediction_output.groupby("hr", as_index=False).agg(
        MAE=("erreur_absolue", "mean"),
        RMSE=("erreur_carree", lambda values: np.sqrt(np.mean(values))),
        Nombre_observations=("erreur_absolue", "size"),
    )
    hourly_errors.to_csv(BASE_DIR / "erreurs_par_heure.csv", index=False)

    grouped = add_time_groups(prediction_output)
    grouped_errors = grouped.groupby("tranche", observed=True).agg(
        MAE=("erreur_absolue", "mean"),
        RMSE=("erreur_carree", lambda values: np.sqrt(np.mean(values))),
        Nombre_observations=("erreur_absolue", "size"),
    ).reset_index()
    grouped_errors.to_csv(BASE_DIR / "erreurs_par_tranche.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(hourly_errors["hr"], hourly_errors["MAE"], color="#4C78A8")
    plt.xlabel("Heure de la journee")
    plt.ylabel("MAE")
    plt.title("Erreur absolue moyenne par heure")
    plt.xticks(range(24))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "erreur_mae_par_heure.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(grouped_errors["tranche"], grouped_errors["MAE"],
            color=["#72B7B2", "#F2CF5B", "#4C78A8", "#E45756", "#B279A2"])
    plt.xlabel("Tranche horaire")
    plt.ylabel("MAE")
    plt.title("Erreur moyenne par tranche horaire")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "erreur_mae_par_tranche.png", dpi=150)
    plt.close()

    # Importance des variables pour l'arbre optimise.
    importance = pd.DataFrame({
        "Variable": X_train.columns,
        "Importance": final_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    importance.to_csv(BASE_DIR / "importance_variables.csv", index=False)

    plt.figure(figsize=(10, 6))
    top = importance.head(12)
    plt.barh(top["Variable"][::-1], top["Importance"][::-1], color="#4C78A8")
    plt.xlabel("Importance Gini")
    plt.title("Importance des variables - arbre optimise")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "importance_variables.png", dpi=150)
    plt.close()

    # Bonus d'interpretabilite: petit arbre d'une profondeur de 3.
    shallow_tree = DecisionTreeRegressor(
        max_depth=3, random_state=RANDOM_STATE
    )
    shallow_tree.fit(X_train, y_train)
    plt.figure(figsize=(18, 8))
    plot_tree(
        shallow_tree,
        feature_names=list(X_train.columns),
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Arbre interpretable (profondeur 3)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "arbre_profondeur_3.png", dpi=150)
    plt.close()

    print("\nErreurs par tranche:")
    print(grouped_errors.to_string(index=False))
    print("\nTop variables:")
    print(importance.head(10).to_string(index=False))
    print("\nLivrables crees dans:", BASE_DIR)


if __name__ == "__main__":
    main()
