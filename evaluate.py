"""
evaluate.py
Quantitative evaluation of the retrieval + ranking layers, so you have real,
defensible numbers for your resume and interview instead of made-up ones.

Metrics:
- Retrieval: Precision@K against a synthetic "relevant if cuisine/keyword matches" proxy
- Ranking: AUC-ROC and NDCG@5 (already printed during training in ranking.py)
"""

import pandas as pd
import numpy as np
from retrieval import DishRetriever

TEST_QUERIES = {
    "something spicy and cheap under 200": {"spice_level": ["spicy", "very spicy"], "max_price": 200},
    "healthy light dinner option": {"cuisine": ["Healthy Food"]},
    "cheesy comfort food": {"dish_keyword": "cheese"},
    "biryani under 300 rupees": {"cuisine": ["Biryani"], "max_price": 300},
    "cold coffee and snacks": {"cuisine": ["Beverages", "Fast Food"]},
}


def is_relevant(row, criteria):
    if "spice_level" in criteria and row["spice_level"] not in criteria["spice_level"]:
        return False
    if "max_price" in criteria and row["price"] > criteria["max_price"]:
        return False
    if "cuisine" in criteria and row["cuisine"] not in criteria["cuisine"]:
        return False
    if "dish_keyword" in criteria and criteria["dish_keyword"].lower() not in row["dish_name"].lower():
        return False
    return True


def precision_at_k(retriever, query, criteria, k=10):
    results = retriever.retrieve(query, top_k=k)
    relevant_flags = results.apply(lambda r: is_relevant(r, criteria), axis=1)
    return relevant_flags.mean()


if __name__ == "__main__":
    dishes = pd.read_csv("dishes.csv")
    retriever = DishRetriever(dishes)

    precisions = []
    print(f"{'Query':45s} | Precision@10")
    print("-" * 65)
    for query, criteria in TEST_QUERIES.items():
        p = precision_at_k(retriever, query, criteria, k=10)
        precisions.append(p)
        print(f"{query:45s} | {p:.2f}")

    print("-" * 65)
    print(f"Mean Precision@10 across test queries: {np.mean(precisions):.3f}")
    print("\n(Run ranking.py separately for AUC-ROC and NDCG@5 on the ranking model.)")
