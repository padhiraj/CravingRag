"""
eda.py
Exploratory Data Analysis + a real hypothesis test. This is the part of the
project that specifically demonstrates classical data-scientist skills (as
opposed to just ML-engineering skills) -- descriptive stats, visualization,
and statistical inference, not just model training.

Run with: python eda.py
Outputs: prints summary stats + hypothesis test result, saves plots to eda_plots/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

OUT_DIR = "eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)


def describe_data(dishes, interactions):
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Dishes: {len(dishes)} | Interactions: {len(interactions)} | "
          f"Unique users: {interactions['user_id'].nunique()}")
    print("\nPrice distribution:")
    print(dishes["price"].describe().round(1))
    print("\nOverall click-through rate: {:.2%}".format(interactions["clicked"].mean()))
    print("Overall order-through rate (of clicks): {:.2%}".format(
        interactions.loc[interactions["clicked"] == 1, "ordered"].mean()))
    print("\nCuisine distribution:")
    print(dishes["cuisine"].value_counts())


def plot_price_distribution(dishes, save=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(dishes["price"], bins=20, kde=True, ax=ax)
    ax.set_title("Dish Price Distribution")
    ax.set_xlabel("Price (₹)")
    if save:
        fig.savefig(f"{OUT_DIR}/price_distribution.png", dpi=100, bbox_inches="tight")
    return fig


def plot_ctr_by_cuisine(dishes, interactions, save=True):
    merged = interactions.merge(dishes[["dish_id", "cuisine"]], on="dish_id", how="left")
    ctr_by_cuisine = merged.groupby("cuisine")["clicked"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=ctr_by_cuisine.values, y=ctr_by_cuisine.index, orient="h", ax=ax)
    ax.set_title("Click-Through Rate by Cuisine")
    ax.set_xlabel("CTR")
    if save:
        fig.savefig(f"{OUT_DIR}/ctr_by_cuisine.png", dpi=100, bbox_inches="tight")
    return fig, ctr_by_cuisine


def plot_rating_vs_ctr(dishes, interactions, save=True):
    merged = interactions.merge(dishes[["dish_id", "rating"]], on="dish_id", how="left")
    merged["rating_bucket"] = pd.cut(merged["rating"], bins=[3.0, 3.5, 4.0, 4.5, 5.0])
    ctr_by_rating = merged.groupby("rating_bucket", observed=True)["clicked"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ctr_by_rating.plot(kind="bar", ax=ax)
    ax.set_title("Click-Through Rate by Rating Bucket")
    ax.set_ylabel("CTR")
    ax.tick_params(axis="x", rotation=0)
    if save:
        fig.savefig(f"{OUT_DIR}/ctr_by_rating.png", dpi=100, bbox_inches="tight")
    return fig


def hypothesis_test_spice_match(interactions, dishes):
    """
    Real hypothesis test: does matching the user's preferred spice level actually
    increase click-through rate, or could the observed difference be due to chance?

    H0: CTR is the same regardless of spice match
    H1: CTR is higher when spice level matches user preference

    Uses a two-proportion z-test (appropriate for comparing two binary conversion
    rates -- exactly the kind of test used for A/B testing in industry).
    """
    merged = interactions.merge(dishes[["dish_id", "spice_level"]], on="dish_id", how="left")
    merged["spice_match"] = merged["spice_level"] == merged["user_pref_spice"]

    matched = merged[merged["spice_match"]]["clicked"]
    unmatched = merged[~merged["spice_match"]]["clicked"]

    n1, n2 = len(matched), len(unmatched)
    p1, p2 = matched.mean(), unmatched.mean()
    p_pool = (matched.sum() + unmatched.sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST: Does spice-level match increase click-through rate?")
    print("=" * 70)
    print(f"CTR when spice matches:     {p1:.2%}  (n={n1})")
    print(f"CTR when spice doesn't match: {p2:.2%}  (n={n2})")
    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value: {p_value:.6f}")
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: statistically significant at alpha={alpha} -- reject H0. "
              f"Spice-matching is a genuine driver of engagement, not noise.")
    else:
        print(f"Result: not statistically significant at alpha={alpha} -- fail to reject H0.")
    return z_stat, p_value


if __name__ == "__main__":
    dishes = pd.read_csv("dishes.csv")
    interactions = pd.read_csv("interactions.csv")

    describe_data(dishes, interactions)
    plot_price_distribution(dishes)
    plt.close("all")
    fig, ctr_by_cuisine = plot_ctr_by_cuisine(dishes, interactions)
    plt.close("all")
    plot_rating_vs_ctr(dishes, interactions)
    plt.close("all")
    hypothesis_test_spice_match(interactions, dishes)

    print(f"\nPlots saved to ./{OUT_DIR}/")
