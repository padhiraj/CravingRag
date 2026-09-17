"""
pipeline.py
Ties together Retrieval -> Ranking -> Generation into a single callable function.
This is the module the Streamlit app (and your interview whiteboard explanation) calls.
"""

import pandas as pd
import joblib

from retrieval import DishRetriever
from generation import generate_explanations

FEATURE_COLS = ["rating", "price", "delivery_time_min",
                 "price_diff_from_budget", "spice_match", "retrieval_score"]


class SwiggyMindPipeline:
    def __init__(self, dishes_csv="dishes.csv", model_path="ranker_model.pkl"):
        self.dishes = pd.read_csv(dishes_csv)
        self.retriever = DishRetriever(self.dishes)
        self.ranker = joblib.load(model_path)

    def recommend(self, query: str, user_pref_budget: int = 300,
                   user_pref_spice: str = "medium", top_k_retrieve: int = 20,
                   top_n_final: int = 5, use_llm: bool = None) -> pd.DataFrame:

        # 1. RETRIEVAL
        candidates = self.retriever.retrieve(query, top_k=top_k_retrieve)

        # 2. RANKING
        candidates["price_diff_from_budget"] = user_pref_budget - candidates["price"]
        candidates["spice_match"] = (candidates["spice_level"] == user_pref_spice).astype(int)
        scores = self.ranker.predict_proba(candidates[FEATURE_COLS])[:, 1]
        candidates["rank_score"] = scores
        ranked = candidates.sort_values("rank_score", ascending=False).head(top_n_final).reset_index(drop=True)

        # 3. GENERATION
        final = generate_explanations(ranked, query, use_llm=use_llm)
        return final[["dish_name", "restaurant_name", "price", "rating",
                       "spice_level", "delivery_time_min", "retrieval_score",
                       "rank_score", "explanation"]]


if __name__ == "__main__":
    pipeline = SwiggyMindPipeline()
    results = pipeline.recommend(
        query="something spicy and cheap under 200",
        user_pref_budget=200,
        user_pref_spice="spicy",
    )
    pd.set_option("display.width", 150)
    print(results)
