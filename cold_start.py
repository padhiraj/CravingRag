"""
cold_start.py
Explicit handling for the cold-start problem: what happens when a brand-new
dish/restaurant has zero interaction history? This is the single most common
follow-up question on any recommender-systems project, so it gets handled as
a first-class code path here rather than an answer you have to improvise.

Strategy implemented (content-based fallback):
  - New items still go through retrieval normally (embeddings need no interaction
    history -- they're built from the item's description text alone).
  - For ranking, a brand-new item has no logged click history, but our feature
    set (rating, price, spice, retrieval_score) is entirely CONTENT-based, not
    collaborative -- meaning the ranking model can score a new item exactly as
    well as an old one, as long as it has basic metadata (rating, price, cuisine).
  - The only real gap: a restaurant with a genuinely brand-new, un-rated item
    has no `rating` yet. For that specific case, we fall back to the
    restaurant's average existing rating, and slightly down-weight the item's
    rank_score to reflect our uncertainty (an "exploration" adjustment), so it
    still gets shown but doesn't unfairly dominate a well-established dish.
"""

import pandas as pd
import numpy as np
import joblib

from retrieval import DishRetriever

FEATURE_COLS = ["rating", "price", "delivery_time_min",
                 "price_diff_from_budget", "spice_match", "retrieval_score"]

EXPLORATION_PENALTY = 0.9  # shrink confidence for items with no track record


def add_new_dish(dishes_df: pd.DataFrame, restaurant_id: int, dish_name: str,
                  cuisine: str, price: int, spice_level: str, dietary: str,
                  restaurant_name: str, city: str, delivery_time_min: int) -> pd.DataFrame:
    """Simulates a restaurant adding a brand-new dish with zero interaction history."""
    existing_restaurant_dishes = dishes_df[dishes_df["restaurant_id"] == restaurant_id]
    if len(existing_restaurant_dishes) > 0:
        # cold start fallback: use the restaurant's existing average rating
        fallback_rating = existing_restaurant_dishes["rating"].mean()
    else:
        # fully new restaurant too -- fall back to city/cuisine average
        similar = dishes_df[dishes_df["cuisine"] == cuisine]
        fallback_rating = similar["rating"].mean() if len(similar) > 0 else dishes_df["rating"].mean()

    new_row = {
        "dish_id": dishes_df["dish_id"].max() + 1,
        "restaurant_id": restaurant_id,
        "restaurant_name": restaurant_name,
        "dish_name": dish_name,
        "cuisine": cuisine,
        "price": price,
        "spice_level": spice_level,
        "dietary": dietary,
        "rating": round(fallback_rating, 1),
        "city": city,
        "delivery_time_min": delivery_time_min,
        "description": (
            f"{dish_name} - a {spice_level} {cuisine.lower()} dish, {dietary}, "
            f"priced at Rs.{price}. Served at {restaurant_name} in {city}, "
            f"rated {round(fallback_rating, 1)} stars (new item, rating estimated "
            f"from restaurant/cuisine average)."
        ),
        "is_cold_start": True,
    }
    return pd.concat([dishes_df, pd.DataFrame([new_row])], ignore_index=True)


def rank_with_cold_start_handling(model, candidates: pd.DataFrame,
                                    user_pref_budget: int, user_pref_spice: str) -> pd.DataFrame:
    df = candidates.copy()
    if "is_cold_start" not in df.columns:
        df["is_cold_start"] = False
    df["is_cold_start"] = df["is_cold_start"].fillna(False)

    df["price_diff_from_budget"] = user_pref_budget - df["price"]
    df["spice_match"] = (df["spice_level"] == user_pref_spice).astype(int)

    scores = model.predict_proba(df[FEATURE_COLS])[:, 1]
    df["rank_score"] = scores

    # exploration penalty: shrink (not zero out) confidence for untested items,
    # so they still surface for user feedback but don't dominate the top slot
    # purely off an estimated rating.
    df.loc[df["is_cold_start"], "rank_score"] *= EXPLORATION_PENALTY

    return df.sort_values("rank_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    dishes = pd.read_csv("dishes.csv")
    dishes["is_cold_start"] = False
    model = joblib.load("ranker_model.pkl")

    # Simulate a restaurant launching a brand-new dish today
    dishes_with_new = add_new_dish(
        dishes, restaurant_id=1, dish_name="Fiery Ghost Pepper Wings",
        cuisine="Fast Food", price=249, spice_level="very spicy", dietary="non-veg",
        restaurant_name="Royal Bites #1", city="Delhi", delivery_time_min=40,
    )
    print(f"Added cold-start item: {dishes_with_new.iloc[-1]['dish_name']} "
          f"(estimated rating: {dishes_with_new.iloc[-1]['rating']})")

    retriever = DishRetriever(dishes_with_new)
    candidates = retriever.retrieve("something very spicy", top_k=20)
    candidates = candidates.merge(
        dishes_with_new[["dish_id", "is_cold_start"]], on="dish_id", how="left"
    )

    ranked = rank_with_cold_start_handling(model, candidates, user_pref_budget=300,
                                             user_pref_spice="very spicy")
    print("\nTop 5 results (cold-start item flagged with [NEW]):")
    for _, row in ranked.head(5).iterrows():
        tag = " [NEW]" if row["is_cold_start"] else ""
        print(f"  {row['dish_name']}{tag} — rank_score={row['rank_score']:.3f}")
