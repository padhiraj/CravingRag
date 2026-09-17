"""
generation.py
The "G" in RAG. Takes the top-ranked, retrieved candidates (with their real
metadata) and generates a short natural-language explanation for each one.

Uses the Anthropic API if ANTHROPIC_API_KEY is set in the environment.
Falls back to a deterministic template-based explanation otherwise, so the
whole pipeline still runs end-to-end without any API key -- useful for a demo
or if you're rate-limited.

KEY DESIGN POINT FOR INTERVIEWS:
The explanation is grounded -- the prompt only gives the model facts that are
already in your retrieved/ranked data (rating, price, spice, distance). This
is what "grounding" means in RAG: the model can't invent facts not present in
the context, which is exactly what you'd be asked to explain if asked
"how do you prevent hallucination here?"
"""

import os
import pandas as pd


def _template_explanation(row: pd.Series) -> str:
    reasons = []
    if row["rating"] >= 4.3:
        reasons.append(f"highly rated ({row['rating']}★)")
    if row.get("spice_match", 0) == 1:
        reasons.append(f"matches your preferred {row['spice_level']} spice level")
    if row.get("price_diff_from_budget", 0) >= 0:
        reasons.append(f"fits your budget at ₹{row['price']}")
    if row["delivery_time_min"] <= 30:
        reasons.append(f"quick delivery ({row['delivery_time_min']} min)")
    if not reasons:
        reasons.append(f"a good semantic match for your query")
    return f"{row['dish_name']} from {row['restaurant_name']} — " + ", ".join(reasons) + "."


def _llm_explanation(row: pd.Series, query: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    facts = (
        f"Dish: {row['dish_name']}\n"
        f"Restaurant: {row['restaurant_name']}\n"
        f"Price: Rs.{row['price']}\n"
        f"Rating: {row['rating']}\n"
        f"Spice level: {row['spice_level']}\n"
        f"Delivery time: {row['delivery_time_min']} min\n"
    )
    prompt = (
        f"A user searched for: \"{query}\"\n\n"
        f"Here is one recommended dish, with ONLY these verified facts:\n{facts}\n\n"
        "Write ONE short, friendly sentence (max 20 words) explaining why this "
        "dish suits the user's query. Use ONLY the facts given above -- do not "
        "invent any detail not listed."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def generate_explanations(ranked_df: pd.DataFrame, query: str, use_llm: bool = None) -> pd.DataFrame:
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    explanations = []
    for _, row in ranked_df.iterrows():
        if use_llm:
            try:
                explanations.append(_llm_explanation(row, query))
                continue
            except Exception as e:
                print(f"[generation] LLM call failed ({e}), falling back to template.")
        explanations.append(_template_explanation(row))

    ranked_df = ranked_df.copy()
    ranked_df["explanation"] = explanations
    return ranked_df


if __name__ == "__main__":
    dishes = pd.read_csv("dishes.csv").head(3).copy()
    dishes["spice_match"] = 1
    dishes["price_diff_from_budget"] = 50
    result = generate_explanations(dishes, "something spicy and cheap")
    for _, r in result.iterrows():
        print("-", r["explanation"])
