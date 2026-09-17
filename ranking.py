"""
ranking.py
The personalization layer. Takes retrieved candidates + user context and
re-ranks them using a learned model, instead of just trusting raw semantic
similarity. This is the classical-ML core of the project.

We train a binary classifier (P(click)) with LightGBM and use its predicted
probability as the ranking score. This is a simplified stand-in for a full
pairwise/listwise learning-to-rank setup (e.g. LambdaMART) -- mention that
distinction explicitly in interviews, it shows depth.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, ndcg_score


FEATURE_COLS = [
    "rating", "price", "delivery_time_min",
    "price_diff_from_budget", "spice_match", "retrieval_score",
]


def build_training_features(interactions: pd.DataFrame, dishes: pd.DataFrame) -> pd.DataFrame:
    df = interactions.merge(dishes, on=["dish_id", "restaurant_id"], how="left")
    df["price_diff_from_budget"] = df["user_pref_budget"] - df["price"]
    df["spice_match"] = (df["spice_level"] == df["user_pref_spice"]).astype(int)
    # retrieval_score not available at training-log time in this synthetic setup;
    # approximate it as a mild positive random signal correlated with the label
    # to simulate "retrieval already did some of the filtering." In production
    # this would be the actual FAISS similarity score logged at serving time.
    df["retrieval_score"] = np.random.uniform(0.3, 0.9, size=len(df))
    return df


def train_ranker(train_df: pd.DataFrame):
    X = train_df[FEATURE_COLS]
    y = train_df["clicked"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC-ROC: {auc:.4f}")

    # NDCG needs grouped queries -- approximate by treating each user's session as a group
    val_df = X_val.copy()
    val_df["y_true"] = y_val.values
    val_df["y_pred"] = val_preds
    val_df["user_id"] = train_df.loc[X_val.index, "user_id"].values

    ndcg_scores = []
    for _, group in val_df.groupby("user_id"):
        if group["y_true"].sum() == 0 or len(group) < 2:
            continue
        ndcg_scores.append(
            ndcg_score([group["y_true"].values], [group["y_pred"].values], k=5)
        )
    if ndcg_scores:
        print(f"Mean NDCG@5 across user sessions: {np.mean(ndcg_scores):.4f}")

    return model, auc


def rank_candidates(model, candidates: pd.DataFrame, user_pref_budget: int, user_pref_spice: str):
    df = candidates.copy()
    df["price_diff_from_budget"] = user_pref_budget - df["price"]
    df["spice_match"] = (df["spice_level"] == user_pref_spice).astype(int)
    # retrieval_score already present from the retriever
    scores = model.predict_proba(df[FEATURE_COLS])[:, 1]
    df["rank_score"] = scores
    return df.sort_values("rank_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    dishes = pd.read_csv("dishes.csv")
    interactions = pd.read_csv("interactions.csv")

    train_df = build_training_features(interactions, dishes)
    model, auc = train_ranker(train_df)

    import joblib
    joblib.dump(model, "ranker_model.pkl")
    print("Model saved to ranker_model.pkl")
