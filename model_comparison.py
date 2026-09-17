"""
model_comparison.py
Compares the LightGBM ranker against two baselines, and explains the winning
model's predictions with SHAP. This is the piece that demonstrates rigor:
never present one model's numbers without showing what "doing nothing clever"
would have scored, and never ship a model to a stakeholder without being able
to explain WHY it makes the predictions it does.

Baselines:
  1. Random          -- lower bound, sanity check
  2. Popularity-only  -- rank purely by rating (a common, deceptively strong baseline)
  3. Logistic Regression -- a simple, interpretable learned baseline
  4. LightGBM (the project's actual model)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ranking import build_training_features, train_ranker, FEATURE_COLS


def evaluate_random_baseline(y_val):
    rng = np.random.default_rng(42)
    random_preds = rng.random(len(y_val))
    return roc_auc_score(y_val, random_preds)


def evaluate_popularity_baseline(X_val, y_val):
    # "popularity" baseline: just use rating as the score, nothing learned
    return roc_auc_score(y_val, X_val["rating"])


def evaluate_logistic_regression(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    preds = model.predict_proba(X_val_scaled)[:, 1]
    return roc_auc_score(y_val, preds), model


def run_comparison(verbose=True):
    dishes = pd.read_csv("dishes.csv")
    interactions = pd.read_csv("interactions.csv")
    train_df = build_training_features(interactions, dishes)

    X = train_df[FEATURE_COLS]
    y = train_df["clicked"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    results["Random baseline"] = evaluate_random_baseline(y_val)
    results["Popularity baseline (rating only)"] = evaluate_popularity_baseline(X_val, y_val)
    logreg_auc, logreg_model = evaluate_logistic_regression(X_train, y_train, X_val, y_val)
    results["Logistic Regression"] = logreg_auc

    if verbose:
        print("Training LightGBM (this also prints its own AUC/NDCG)...")
    lgbm_model, lgbm_auc = train_ranker(train_df)
    results["LightGBM (final model)"] = lgbm_auc

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON — Validation AUC-ROC")
        print("=" * 60)
        for name, auc in sorted(results.items(), key=lambda x: x[1]):
            bar = "█" * int(auc * 40)
            print(f"{name:38s} {auc:.4f}  {bar}")

    return lgbm_model, X_val, logreg_model, results


def get_shap_importance(model, X_val):
    """Returns a pandas Series of mean |SHAP value| per feature, sorted descending."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs_shap, index=X_val.columns).sort_values(ascending=False)


def explain_with_shap(model, X_val):
    import shap
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (SHAP values on LightGBM model)")
    print("=" * 60)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Handle both binary-classifier output shapes across shap/lightgbm versions
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_val.columns).sort_values(ascending=False)

    print("\nMean |SHAP value| per feature (higher = more influence on click prediction):")
    for feature, value in importance.items():
        bar = "█" * int(value / importance.max() * 40)
        print(f"{feature:28s} {value:.4f}  {bar}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X_val, show=False)
    plt.savefig("eda_plots/shap_summary.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("\nSHAP summary plot saved to eda_plots/shap_summary.png")


if __name__ == "__main__":
    import os
    os.makedirs("eda_plots", exist_ok=True)
    lgbm_model, X_val, logreg_model, results = run_comparison()
    explain_with_shap(lgbm_model, X_val)
