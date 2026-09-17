"""
app.py
Streamlit dashboard for SwiggyMind: a live recommendation demo PLUS the
data-science analysis behind it (EDA, model comparison, explainability,
A/B test results, cold-start handling) -- so the whole project can be
demoed in an interview from one screen instead of scattered terminal output.

Run with: streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import SwiggyMindPipeline
from retrieval import DishRetriever
import eda
import model_comparison
import ab_testing
import cold_start

st.set_page_config(page_title="SwiggyMind Dashboard", page_icon="🍽️", layout="wide")

st.title("🍽️ CravingRag")
st.caption("Conversational Food Discovery Engine — RAG + Learned Ranking + Full DS Analysis")


# ---------- Cached data / model loading (so switching tabs doesn't retrain everything) ----------

@st.cache_resource
def load_pipeline():
    return SwiggyMindPipeline()


@st.cache_data
def load_raw_data():
    return pd.read_csv("dishes.csv"), pd.read_csv("interactions.csv")


@st.cache_data
def compute_eda(_dishes, _interactions):
    ctr_fig, ctr_by_cuisine = eda.plot_ctr_by_cuisine(_dishes, _interactions, save=False)
    price_fig = eda.plot_price_distribution(_dishes, save=False)
    rating_fig = eda.plot_rating_vs_ctr(_dishes, _interactions, save=False)
    z_stat, p_value = eda.hypothesis_test_spice_match(_interactions, _dishes)
    return {
        "ctr_fig": ctr_fig, "ctr_by_cuisine": ctr_by_cuisine,
        "price_fig": price_fig, "rating_fig": rating_fig,
        "z_stat": z_stat, "p_value": p_value,
    }


@st.cache_resource
def compute_model_comparison():
    lgbm_model, X_val, logreg_model, results = model_comparison.run_comparison(verbose=False)
    shap_importance = model_comparison.get_shap_importance(lgbm_model, X_val)
    return lgbm_model, X_val, results, shap_importance


@st.cache_data
def compute_ab_test(n_users=800):
    n_needed = ab_testing.required_sample_size(baseline_rate=0.20, mde=0.05, alpha=0.05, power=0.8)
    control, treatment = ab_testing.simulate_ab_test(n_users=n_users)
    return n_needed, control, treatment


pipeline = load_pipeline()
dishes, interactions = load_raw_data()

tab_demo, tab_eda, tab_model, tab_ab, tab_cold = st.tabs(
    ["🔍 Live Demo", "📊 Data Insights", "🤖 Model Performance", "🧪 A/B Test", "🆕 Cold Start"]
)

# ---------------------------------------------------------------- TAB 1: Live Demo
with tab_demo:
    with st.sidebar:
        st.header("Your Preferences")
        budget = st.slider("Budget (₹ for one item)", 100, 600, 250, step=25)
        spice = st.selectbox("Preferred spice level", ["mild", "medium", "spicy", "very spicy"])
        top_n = st.slider("Number of recommendations", 3, 10, 5)
        st.markdown("---")
        st.caption(
            "Architecture: Query → TF-IDF/SVD embedding → cosine similarity retrieval → "
            "LightGBM re-ranking → grounded explanation generation."
        )

    query = st.text_input(
        "What are you craving?",
        placeholder="e.g. something spicy and cheap under 200",
    )

    if st.button("Find food") and query:
        with st.spinner("Retrieving, ranking, and explaining..."):
            results = pipeline.recommend(
                query=query, user_pref_budget=budget, user_pref_spice=spice, top_n_final=top_n,
            )
        st.subheader(f'Top {len(results)} picks for: "{query}"')
        for _, row in results.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{row['dish_name']}** — *{row['restaurant_name']}*")
                    st.write(row["explanation"])
                with col2:
                    st.metric("Price", f"₹{row['price']}")
                    st.caption(f"⭐ {row['rating']}  |  🕒 {row['delivery_time_min']} min")
                st.progress(min(float(row["rank_score"]), 1.0),
                            text=f"Rank confidence: {row['rank_score']:.2f}")
    else:
        st.info("Enter a craving above and hit **Find food** to see the pipeline run live.")

# ---------------------------------------------------------------- TAB 2: EDA
with tab_eda:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dishes", f"{len(dishes):,}")
    c2.metric("Interactions", f"{len(interactions):,}")
    c3.metric("Unique users", f"{interactions['user_id'].nunique():,}")
    c4.metric("Overall CTR", f"{interactions['clicked'].mean():.1%}")

    with st.spinner("Running EDA..."):
        eda_results = compute_eda(dishes, interactions)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(eda_results["price_fig"])
    with col2:
        st.pyplot(eda_results["rating_fig"])
    st.pyplot(eda_results["ctr_fig"])

    st.subheader("Hypothesis Test: Does spice-level match drive clicks?")
    st.markdown(
        "**H₀:** click-through rate is the same regardless of spice match. "
        "**H₁:** CTR is higher when spice level matches user preference. "
        "Tested with a two-proportion z-test (same test family used for real A/B analysis)."
    )
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("Z-statistic", f"{eda_results['z_stat']:.2f}")
    hc2.metric("P-value", f"{eda_results['p_value']:.2e}")
    verdict = "✅ Statistically significant" if eda_results["p_value"] < 0.05 else "❌ Not significant"
    hc3.metric("Verdict (α=0.05)", verdict)

# ---------------------------------------------------------------- TAB 3: Model Performance
with tab_model:
    st.subheader("Model Comparison — Validation AUC-ROC")
    st.caption("A single model's score means nothing without a baseline to compare it against.")
    with st.spinner("Training models and computing SHAP values (first load only)..."):
        lgbm_model, X_val, comp_results, shap_importance = compute_model_comparison()

    comp_df = pd.DataFrame(
        {"Model": list(comp_results.keys()), "AUC-ROC": list(comp_results.values())}
    ).sort_values("AUC-ROC")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(comp_df["Model"], comp_df["AUC-ROC"], color="#ff6b35")
    ax.set_xlabel("AUC-ROC")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    st.subheader("Feature Importance (SHAP)")
    st.caption(
        "SHAP shows the real magnitude/direction of each feature's effect on individual "
        "predictions — more reliable than LightGBM's built-in split-count importance."
    )
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(shap_importance.index[::-1], shap_importance.values[::-1], color="#2e86ab")
    ax2.set_xlabel("Mean |SHAP value|")
    st.pyplot(fig2)

# ---------------------------------------------------------------- TAB 4: A/B Test
with tab_ab:
    st.subheader("Simulated A/B Test: Learned Ranking vs Raw Retrieval")
    st.markdown(
        "**Control:** results ordered by raw retrieval score only (no personalization). "
        "**Treatment:** results re-ranked by the trained LightGBM model."
    )

    with st.spinner("Running power analysis and simulated experiment..."):
        n_needed, control, treatment = compute_ab_test(n_users=800)

    st.info(f"📐 **Pre-experiment power analysis:** to reliably detect a 5-point lift over a "
            f"20% baseline conversion rate (α=0.05, power=0.8), you'd need ~**{n_needed} users per group**.")

    conv1, conv2 = control.mean(), treatment.mean()
    from statsmodels.stats.proportion import proportions_ztest
    count = np.array([control.sum(), treatment.sum()])
    nobs = np.array([len(control), len(treatment)])
    z_stat, p_value = proportions_ztest(count, nobs, alternative="two-sided")

    c1, c2, c3 = st.columns(3)
    c1.metric("Control conversion", f"{conv1:.1%}", help=f"n={len(control)}")
    c2.metric("Treatment conversion", f"{conv2:.1%}", f"{(conv2-conv1)*100:+.1f} pts", help=f"n={len(treatment)}")
    c3.metric("P-value", f"{p_value:.2e}", "significant ✅" if p_value < 0.05 else "not significant ❌")

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.bar(["Control", "Treatment"], [conv1, conv2], color=["#888888", "#ff6b35"])
    ax3.set_ylabel("Conversion rate")
    ax3.set_ylim(0, 1)
    st.pyplot(fig3)

# ---------------------------------------------------------------- TAB 5: Cold Start
with tab_cold:
    st.subheader("Cold-Start Handling: Recommending a Brand-New Dish")
    st.markdown(
        "New items have zero interaction history. This demo adds a brand-new dish and shows "
        "it still gets a fair, content-based ranking — with a small exploration penalty so it "
        "doesn't unfairly dominate proven items purely off an estimated rating."
    )

    with st.form("cold_start_form"):
        c1, c2, c3 = st.columns(3)
        new_dish_name = c1.text_input("New dish name", "Fiery Ghost Pepper Wings")
        new_price = c2.number_input("Price (₹)", 50, 999, 249)
        new_spice = c3.selectbox("Spice level", ["mild", "medium", "spicy", "very spicy"], index=3)
        search_query = st.text_input("Query to test against", "something very spicy")
        submitted = st.form_submit_button("Add dish & rank")

    if submitted:
        dishes_cs = dishes.copy()
        dishes_cs["is_cold_start"] = False
        dishes_with_new = cold_start.add_new_dish(
            dishes_cs, restaurant_id=1, dish_name=new_dish_name, cuisine="Fast Food",
            price=new_price, spice_level=new_spice, dietary="non-veg",
            restaurant_name="Royal Bites #1", city="Delhi", delivery_time_min=40,
        )
        retriever = DishRetriever(dishes_with_new)
        candidates = retriever.retrieve(search_query, top_k=20)
        candidates = candidates.merge(
            dishes_with_new[["dish_id", "is_cold_start"]], on="dish_id", how="left"
        )
        model_for_cs = joblib.load("ranker_model.pkl")
        ranked = cold_start.rank_with_cold_start_handling(
            model_for_cs, candidates, user_pref_budget=budget if 'budget' in dir() else 300,
            user_pref_spice=new_spice,
        )
        st.write(f"Rank position of the new dish out of {len(ranked)} candidates:")
        new_dish_rank = ranked.index[ranked["is_cold_start"] == True]
        position = (new_dish_rank[0] + 1) if len(new_dish_rank) > 0 else "not in top candidates"
        st.metric("New dish rank position", position)

        for i, row in ranked.head(8).iterrows():
            tag = " 🆕" if row["is_cold_start"] else ""
            st.write(f"{i+1}. **{row['dish_name']}**{tag} — rank_score={row['rank_score']:.3f}")
