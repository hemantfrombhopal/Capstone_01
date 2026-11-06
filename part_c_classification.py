"""Part C — Delay Classifier and Alert Threshold."""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.shipment_prep import FEATURES, NOISE_COLUMNS

warnings.filterwarnings("ignore")


def run_part_c():
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # 1. Stratified split (70% / 15% / 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 2. Baseline and Candidate Models on Validation
    models_rows = []

    # Blind baseline: predict 0 (on-time) for all
    blind_preds = np.zeros(len(y_val), dtype=int)
    tn = int((y_val == 0).sum())
    fp = 0
    fn = int((y_val == 1).sum())
    tp = 0
    acc = float((tp + tn) / len(y_val))
    models_rows.append({
        "model": "blind_all_on_time",
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": round(acc, 4),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    })

    # Logistic Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, lr_preds).ravel()
    models_rows.append({
        "model": "logistic_regression",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_val, lr_preds), 4),
        "precision": round(precision_score(y_val, lr_preds, zero_division=0), 4),
        "recall": round(recall_score(y_val, lr_preds), 4),
        "f1": round(f1_score(y_val, lr_preds, zero_division=0), 4),
    })

    # Decision Tree depths: 2, 3, 4, 5, 6, 8, 10
    dt_candidates = []
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        p = dt.predict(X_val)
        f = f1_score(y_val, p, zero_division=0)
        dt_candidates.append((d, f, dt, p))

    # Highest validation F1, ties -> smaller d
    best_dt_tuple = max(dt_candidates, key=lambda item: (item[1], -item[0]))
    best_d, _, _, best_dt_preds = best_dt_tuple
    tn, fp, fn, tp = confusion_matrix(y_val, best_dt_preds).ravel()
    models_rows.append({
        "model": f"decision_tree_depth_{best_d}",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_val, best_dt_preds), 4),
        "precision": round(precision_score(y_val, best_dt_preds, zero_division=0), 4),
        "recall": round(recall_score(y_val, best_dt_preds), 4),
        "f1": round(f1_score(y_val, best_dt_preds, zero_division=0), 4),
    })

    # Random Forest Default
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    rf_preds = rf_def.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, rf_preds).ravel()
    models_rows.append({
        "model": "random_forest_default",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_val, rf_preds), 4),
        "precision": round(precision_score(y_val, rf_preds, zero_division=0), 4),
        "recall": round(recall_score(y_val, rf_preds), 4),
        "f1": round(f1_score(y_val, rf_preds, zero_division=0), 4),
    })

    pd.DataFrame(models_rows).to_csv("outputs/part_c_models.csv", index=False)

    # 3. Feature Scores and Noise Check
    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_
    importances = rf_def.feature_importances_

    scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": f_scores,
        "rf_importance": importances,
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
    })
    scores_df["f_rank"] = scores_df["f_score"].rank(ascending=False, method="min").astype(int)
    scores_df = scores_df.sort_values("f_rank")
    scores_df = scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    scores_df["f_score"] = scores_df["f_score"].round(4)
    scores_df["rf_importance"] = scores_df["rf_importance"].round(4)
    scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

    # SelectK experiments: k in 5, 7, 10, 12, 17
    selectk_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([
            ("sel", SelectKBest(f_classif, k=k)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        p = pipe.predict(X_val)
        selectk_rows.append({
            "k": k,
            "val_f1": round(f1_score(y_val, p, zero_division=0), 4),
        })
    pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

    # 4. Tuning with k = 7
    k7_selector = SelectKBest(f_classif, k=7)
    X_train_k7 = k7_selector.fit_transform(X_train, y_train)
    X_val_k7 = k7_selector.transform(X_val)
    X_test_k7 = k7_selector.transform(X_test)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }
    rf_tune_base = RandomForestClassifier(random_state=42)

    # Grid search (27 candidates * 5 folds = 135 fits)
    grid = GridSearchCV(rf_tune_base, param_grid, cv=5)
    grid.fit(X_train_k7, y_train)
    grid_preds = grid.predict(X_val_k7)
    grid_val_f1 = f1_score(y_val, grid_preds, zero_division=0)

    # Randomized search (9 candidates * 5 folds = 45 fits)
    rnd = RandomizedSearchCV(rf_tune_base, param_grid, n_iter=9, cv=5, random_state=42)
    rnd.fit(X_train_k7, y_train)
    rnd_preds = rnd.predict(X_val_k7)
    rnd_val_f1 = f1_score(y_val, rnd_preds, zero_division=0)

    # Default forest on same 7 features
    def_forest_7 = RandomForestClassifier(n_estimators=100, random_state=42)
    def_forest_7.fit(X_train_k7, y_train)
    def_preds = def_forest_7.predict(X_val_k7)
    def_val_f1 = f1_score(y_val, def_preds, zero_division=0)

    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid.best_params_),
            "best_cv_accuracy": round(grid.best_score_, 4),
            "val_f1": round(grid_val_f1, 4),
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rnd.best_params_),
            "best_cv_accuracy": round(rnd.best_score_, 4),
            "val_f1": round(rnd_val_f1, 4),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": round(def_val_f1, 4),
        },
    ]
    pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)

    # 5. Choosing the final configuration
    # Ties -> default_forest_same_7_features, then grid_search, then randomized_search
    priority = {"default_forest_same_7_features": 3, "grid_search": 2, "randomized_search": 1}
    sorted_tuning = sorted(
        tuning_rows,
        key=lambda r: (r["val_f1"], priority[r["search"]]),
        reverse=True,
    )
    winner = sorted_tuning[0]["search"]

    if winner == "default_forest_same_7_features":
        chosen_n_estimators = 100
        chosen_max_depth = None
        chosen_min_samples_leaf = 1
        model_kwargs = {"n_estimators": 100, "random_state": 42}
    elif winner == "grid_search":
        chosen_n_estimators = grid.best_params_["n_estimators"]
        chosen_max_depth = grid.best_params_["max_depth"]
        chosen_min_samples_leaf = grid.best_params_["min_samples_leaf"]
        model_kwargs = {**grid.best_params_, "random_state": 42}
    else:
        chosen_n_estimators = rnd.best_params_["n_estimators"]
        chosen_max_depth = rnd.best_params_["max_depth"]
        chosen_min_samples_leaf = rnd.best_params_["min_samples_leaf"]
        model_kwargs = {**rnd.best_params_, "random_state": 42}

    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(**model_kwargs)),
    ])
    final_pipeline.fit(X_train, y_train)

    # 6. Alert threshold
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_rows = []
    chosen_threshold = 0.1

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds)
        f1 = f1_score(y_val, preds, zero_division=0)
        threshold_rows.append({
            "threshold": round(t, 1),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })
        if rec >= 0.80:
            chosen_threshold = t

    pd.DataFrame(threshold_rows).to_csv("outputs/part_c_thresholds.csv", index=False)

    # 7. One test evaluation
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)

    test_row = [{
        "threshold": round(chosen_threshold, 1),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(test_acc, 4),
        "precision": round(test_prec, 4),
        "recall": round(test_rec, 4),
        "f1": round(test_f1, 4),
    }]
    pd.DataFrame(test_row).to_csv("outputs/part_c_final_test.csv", index=False)

    # Final config
    config_row = [{
        "selector_k": 7,
        "n_estimators": chosen_n_estimators,
        "max_depth": "unlimited" if chosen_max_depth is None else chosen_max_depth,
        "min_samples_leaf": chosen_min_samples_leaf,
        "threshold": round(chosen_threshold, 1),
        "chosen_from": winner,
    }]
    pd.DataFrame(config_row).to_csv("outputs/part_c_final_config.csv", index=False)

    print(f"Part C complete. Winner: {winner}, Threshold: {chosen_threshold}, Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    run_part_c()

