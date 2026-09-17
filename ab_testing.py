"""
ab_testing.py
Simulates an A/B test comparing two ranking strategies:
  Control (A): show results ordered by raw retrieval_score only (no personalization)
  Treatment (B): show results ordered by the learned ranking model

This directly demonstrates experiment design skills: sample size / power
calculation BEFORE running the test, then a proper significance test on the
results -- exactly the workflow a Swiggy DS is expected to know for evaluating
any product change (a new ranking algorithm, a UI change, a pricing test).
"""

import numpy as np
import pandas as pd
import joblib
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

from retrieval import DishRetriever

FEATURE_COLS = ["rating", "price", "delivery_time_min",
                 "price_diff_from_budget", "spice_match", "retrieval_score"]


def required_sample_size(baseline_rate=0.20, mde=0.02, alpha=0.05, power=0.8):
    """
    Standard pre-experiment power analysis: how many users per group do we need
    to reliably detect a `mde` (minimum detectable effect) lift over baseline,
    at a given significance level and power?
    """
    effect_size = (mde) / np.sqrt(baseline_rate * (1 - baseline_rate))
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
    return int(np.ceil(n))


def simulate_ab_test(n_users=400, seed=123):
    """
    Simulates users being randomly bucketed into control/treatment, issuing a
    query, and clicking based on how well the top-3 shown results match their
    preferences. Treatment (ranked by the trained model) should show a real,
    measurable lift over control (ranked by raw retrieval score).
    """
    rng = np.random.default_rng(seed)
    dishes = pd.read_csv("dishes.csv")
    retriever = DishRetriever(dishes)
    ranker = joblib.load("ranker_model.pkl")

    queries = ["something spicy and cheap under 200", "healthy light dinner option",
               "cheesy comfort food", "biryani under 300 rupees", "cold coffee and snacks"]
    spice_options = ["mild", "medium", "spicy", "very spicy"]
    budget_options = [150, 250, 350, 500]

    control_clicks, treatment_clicks = [], []

    for _ in range(n_users):
        group = "control" if rng.random() < 0.5 else "treatment"
        query = rng.choice(queries)
        user_spice = rng.choice(spice_options)
        user_budget = int(rng.choice(budget_options))

        candidates = retriever.retrieve(query, top_k=20)
        candidates["price_diff_from_budget"] = user_budget - candidates["price"]
        candidates["spice_match"] = (candidates["spice_level"] == user_spice).astype(int)

        if group == "control":
            top3 = candidates.sort_values("retrieval_score", ascending=False).head(3)
        else:
            scores = ranker.predict_proba(candidates[FEATURE_COLS])[:, 1]
            candidates["rank_score"] = scores
            top3 = candidates.sort_values("rank_score", ascending=False).head(3)

        # Simulate: user clicks if any of the top 3 matches their spice preference
        # AND is within budget -- a simple, transparent proxy for "found something good"
        clicked = int((
            (top3["spice_level"] == user_spice) & (top3["price"] <= user_budget)
        ).any())

        if group == "control":
            control_clicks.append(clicked)
        else:
            treatment_clicks.append(clicked)

    return np.array(control_clicks), np.array(treatment_clicks)


def analyze_ab_results(control, treatment, alpha=0.05):
    n1, n2 = len(control), len(treatment)
    conv1, conv2 = control.mean(), treatment.mean()

    count = np.array([control.sum(), treatment.sum()])
    nobs = np.array([n1, n2])
    z_stat, p_value = proportions_ztest(count, nobs, alternative="two-sided")

    print("=" * 70)
    print("A/B TEST RESULTS: Control (retrieval-only) vs Treatment (learned ranking)")
    print("=" * 70)
    print(f"Control conversion rate:   {conv1:.2%}  (n={n1})")
    print(f"Treatment conversion rate: {conv2:.2%}  (n={n2})")
    print(f"Absolute lift: {(conv2 - conv1):+.2%}")
    print(f"Relative lift: {((conv2 - conv1) / conv1 * 100 if conv1 > 0 else float('nan')):+.1f}%")
    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < alpha:
        print(f"Result: statistically significant at alpha={alpha}. "
              f"The learned ranking model produces a real improvement over raw retrieval.")
    else:
        print(f"Result: NOT statistically significant at alpha={alpha}. "
              f"Would need a larger sample to draw a confident conclusion.")


if __name__ == "__main__":
    print("--- Pre-experiment power analysis ---")
    n_needed = required_sample_size(baseline_rate=0.20, mde=0.05, alpha=0.05, power=0.8)
    print(f"To detect a 5-point lift over a 20% baseline conversion rate "
          f"(alpha=0.05, power=0.8), you'd need ~{n_needed} users PER GROUP.\n")

    control, treatment = simulate_ab_test(n_users=800)
    analyze_ab_results(control, treatment)
