"""
data_generator.py
Generates a synthetic but realistic restaurant + dish dataset, plus a synthetic
user-interaction log (query -> shown items -> clicked/ordered).

This stands in for scraped/proprietary Swiggy data. Being upfront that this is
synthetic is completely fine in interviews -- what matters is the pipeline design.
"""

import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

CUISINES = ["North Indian", "South Indian", "Chinese", "Italian", "Fast Food",
            "Biryani", "Desserts", "Beverages", "Continental", "Mexican",
            "Bakery", "Healthy Food", "Street Food"]

CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune", "Jamshedpur", "Chennai"]

DISH_ADJECTIVES = ["Spicy", "Creamy", "Tangy", "Crispy", "Loaded", "Classic",
                    "Homestyle", "Fiery", "Cheesy", "Grilled", "Steamed", "Roasted"]

DISH_BASE = {
    "North Indian": ["Butter Chicken", "Paneer Tikka", "Dal Makhani", "Chole Bhature", "Rogan Josh"],
    "South Indian": ["Masala Dosa", "Idli Sambar", "Uttapam", "Filter Coffee", "Curd Rice"],
    "Chinese": ["Hakka Noodles", "Manchurian", "Fried Rice", "Momos", "Spring Rolls"],
    "Italian": ["Margherita Pizza", "Pasta Alfredo", "Lasagna", "Garlic Bread", "Risotto"],
    "Fast Food": ["Cheese Burger", "French Fries", "Chicken Wrap", "Nuggets", "Hot Dog"],
    "Biryani": ["Chicken Biryani", "Mutton Biryani", "Veg Biryani", "Egg Biryani", "Prawn Biryani"],
    "Desserts": ["Gulab Jamun", "Chocolate Brownie", "Ice Cream Sundae", "Rasmalai", "Cheesecake"],
    "Beverages": ["Mango Lassi", "Cold Coffee", "Fresh Lime Soda", "Milkshake", "Iced Tea"],
    "Continental": ["Grilled Chicken", "Caesar Salad", "Steak", "Sandwich", "Soup"],
    "Mexican": ["Tacos", "Burrito Bowl", "Nachos", "Quesadilla", "Enchiladas"],
    "Bakery": ["Croissant", "Muffin", "Donut", "Bagel", "Cinnamon Roll"],
    "Healthy Food": ["Quinoa Salad", "Grilled Tofu Bowl", "Sprout Salad", "Oats Bowl", "Smoothie Bowl"],
    "Street Food": ["Pani Puri", "Vada Pav", "Bhel Puri", "Pav Bhaji", "Kathi Roll"],
}

DIETARY_TAGS = ["veg", "non-veg", "vegan", "jain"]


def generate_restaurants(n=200):
    rows = []
    for i in range(n):
        cuisine_count = random.randint(1, 3)
        cuisines = random.sample(CUISINES, cuisine_count)
        rows.append({
            "restaurant_id": i + 1,
            "name": f"{random.choice(['The', 'Royal', 'Spice', 'Urban', 'Cafe', 'Grand'])} "
                    f"{random.choice(['Kitchen', 'Bites', 'House', 'Corner', 'Junction', 'Table'])} #{i+1}",
            "city": random.choice(CITIES),
            "cuisines": ", ".join(cuisines),
            "rating": round(random.uniform(3.0, 4.9), 1),
            "avg_cost_for_two": random.choice([150, 200, 250, 300, 400, 500, 700, 900]),
            "delivery_time_min": random.randint(15, 55),
        })
    return pd.DataFrame(rows)


def generate_dishes(restaurants_df, dishes_per_restaurant=4):
    rows = []
    dish_id = 1
    for _, r in restaurants_df.iterrows():
        cuisines = r["cuisines"].split(", ")
        for _ in range(dishes_per_restaurant):
            cuisine = random.choice(cuisines)
            base_dish = random.choice(DISH_BASE[cuisine])
            adjective = random.choice(DISH_ADJECTIVES)
            dish_name = f"{adjective} {base_dish}"
            price = random.choice([99, 129, 149, 179, 199, 249, 299, 349, 399])
            spice_level = random.choice(["mild", "medium", "spicy", "very spicy"])
            dietary = random.choice(DIETARY_TAGS)
            description = (
                f"{dish_name} - a {spice_level} {cuisine.lower()} dish, {dietary}, "
                f"priced at Rs.{price}. Served at {r['name']} in {r['city']}, "
                f"rated {r['rating']} stars."
            )
            rows.append({
                "dish_id": dish_id,
                "restaurant_id": r["restaurant_id"],
                "restaurant_name": r["name"],
                "dish_name": dish_name,
                "cuisine": cuisine,
                "price": price,
                "spice_level": spice_level,
                "dietary": dietary,
                "rating": r["rating"],
                "city": r["city"],
                "delivery_time_min": r["delivery_time_min"],
                "description": description,
            })
            dish_id += 1
    return pd.DataFrame(rows)


SAMPLE_QUERIES = [
    "something spicy and cheap under 200",
    "healthy light dinner option",
    "cheesy comfort food",
    "quick breakfast idea",
    "sweet dessert to end the meal",
    "vegan friendly dinner",
    "biryani under 300 rupees",
    "cold coffee and snacks",
    "mild non spicy food for kids",
    "street food cravings",
]


def generate_interaction_log(dishes_df, n_users=150, interactions_per_user=8):
    """
    Simulates: a user issues a query, N dishes are shown (retrieved), user clicks/orders some.
    Click/order probability is made to correlate with rating, price fit, and spice match --
    this gives the ranking model real signal to learn from instead of pure noise.
    """
    rows = []
    for user_id in range(1, n_users + 1):
        user_pref_spice = random.choice(["mild", "medium", "spicy", "very spicy"])
        user_pref_budget = random.choice([150, 250, 350, 500])
        for _ in range(interactions_per_user):
            query = random.choice(SAMPLE_QUERIES)
            shown = dishes_df.sample(10, random_state=random.randint(0, 100000))
            for _, dish in shown.iterrows():
                score = 0.0
                score += (dish["rating"] - 3.0) / 2.0          # higher rating -> more likely liked
                score += 1.0 if dish["price"] <= user_pref_budget else -0.5
                score += 0.5 if dish["spice_level"] == user_pref_spice else 0.0
                score += np.random.normal(0, 0.4)               # noise
                click_prob = 1 / (1 + np.exp(-score))            # sigmoid
                clicked = np.random.rand() < click_prob
                ordered = clicked and (np.random.rand() < 0.4)
                rows.append({
                    "user_id": user_id,
                    "query": query,
                    "dish_id": dish["dish_id"],
                    "restaurant_id": dish["restaurant_id"],
                    "user_pref_budget": user_pref_budget,
                    "user_pref_spice": user_pref_spice,
                    "clicked": int(clicked),
                    "ordered": int(ordered),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    restaurants = generate_restaurants(200)
    dishes = generate_dishes(restaurants)
    interactions = generate_interaction_log(dishes)

    restaurants.to_csv("restaurants.csv", index=False)
    dishes.to_csv("dishes.csv", index=False)
    interactions.to_csv("interactions.csv", index=False)

    print(f"Generated {len(restaurants)} restaurants, {len(dishes)} dishes, "
          f"{len(interactions)} interaction rows.")
