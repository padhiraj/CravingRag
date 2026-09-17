# SwiggyMind — Conversational Food Discovery Engine

I built this after thinking about how clunky food app search still is. You type "something light, not too oily, under 250 bucks" and most apps just do keyword matching against dish names, so you get garbage results or nothing at all. This project is my attempt at fixing that: a small end-to-end system that takes a natural-language craving, retrieves relevant dishes using semantic search, re-ranks them with a model trained on (simulated) user behavior, and explains *why* each result was picked instead of just listing them.

It's not trying to be a production system — it's a project I built to actually understand how retrieval-augmented generation, learning-to-rank, and recommendation systems fit together, instead of just reading about them separately.

## What it actually does

You type something like `"spicy and cheap under 200"`. Behind the scenes:

1. The query gets embedded and compared against ~800 dish descriptions using cosine similarity (this is the retrieval step — the "R" in RAG)
2. The top candidates get re-ranked by a LightGBM model that's learned what actually gets clicked, based on price fit, rating, spice match, and delivery time
3. The final results get a short, grounded explanation generated for each one — grounded meaning the explanation can only reference facts that are actually true about that dish, so it can't just make things up

There's also a full dashboard (`streamlit run app.py`) that shows the data analysis behind the recommender, not just the recommender itself — EDA, a hypothesis test, model comparison against baselines, SHAP explainability, a simulated A/B test, and a cold-start demo. I added these mostly because I wanted the project to demonstrate actual data science thinking, not just "trained a model, made an app."

## Quick start

```bash

cd swiggymind
python -m venv venv
source venv/bin/activate        
pip install -r requirements.txt

python data_generator.py        
python ranking.py              
streamlit run app.py            
```

If you want to poke around the individual pieces without the UI:
```bash
python eda.py               
python model_comparison.py  
python ab_testing.py        
python cold_start.py        
python pipeline.py          
```

Optional: set `ANTHROPIC_API_KEY` if you want real LLM-generated explanations instead of the template fallback. It works fine without it — I built the fallback specifically so the whole thing runs offline with zero API dependency.

## Architecture

```
query
  │
  ▼
retrieval (TF-IDF + SVD embeddings, cosine similarity)
  │
  ▼
ranking (LightGBM, trained on simulated clicks)
  │
  ▼
generation (LLM explanation, grounded in retrieved facts)
```

I originally tried using FAISS for the retrieval step, but it kept segfaulting on Apple Silicon because of an OpenMP conflict with LightGBM (both bundle their own OpenMP runtime, and loading both in one process is a known problem). Rather than fight a native library issue, I just switched to a plain NumPy matrix multiply for the similarity search. At ~800 dishes that's sub-millisecond anyway — brute force is genuinely the right call at this scale, and it's literally what FAISS's simplest index does internally. If this ever needed to scale to hundreds of thousands of items, that's when I'd actually reach for FAISS or a proper vector DB.

## Some real numbers from running this

These aren't made up — they're what I got running it on my machine (yours will vary slightly since there's randomness in the synthetic data generation):

- Retrieval precision@10 across a handful of test queries: ~0.74
- Ranking model AUC-ROC: ~0.68, NDCG@5: ~0.85
- Baseline comparison: random (0.48) → popularity-only (0.54) → logistic regression (0.67) → LightGBM (0.68) — the model is genuinely earning its complexity, it's not just noise
- SHAP says `price_diff_from_budget` is the single biggest driver of predicted clicks, which honestly makes intuitive sense — people care about hitting their budget more than almost anything else
- A/B test simulation: the learned ranker beats raw retrieval-only ordering by a wide, statistically significant margin (p < 0.0001)

## Things I'd be upfront about if you ask me

- **The data is synthetic.** There's a public Zomato/Swiggy-style restaurant dataset underneath, but the dish-level data and all the click/order interactions are simulated by me with a hand-designed probability function (rating + budget fit + spice match + noise → click probability). I did this deliberately so the model would have real signal to learn from instead of pure randomness, but it's not real user behavior.
- **The ranking model is a binary classifier, not a true learning-to-rank model.** I used LightGBM's classifier on click labels rather than a proper pairwise/listwise ranker like LambdaMART. It works fine for this scale, but a real production system would probably use `LGBMRanker` or something similar.
- **The embeddings are TF-IDF + SVD, not transformer-based.** I kept it dependency-light on purpose. If you want real semantic embeddings, swap in `sentence-transformers` — there's a commented-out drop-in class in `retrieval.py` for exactly this.

## What I'd do differently with more time

- Actually fine-tune an embedding model on query-to-dish click pairs instead of using generic TF-IDF
- Move to a proper listwise ranking loss instead of binary classification
- Add real geographic/distance-based filtering (right now delivery time is just a static per-restaurant number, not computed from actual location)
- Log real interaction data from an actual small user test instead of relying on simulated clicks

## Repo structure

```
data_generator.py     synthetic restaurants/dishes/interactions
eda.py                exploratory analysis + hypothesis test
retrieval.py           embedding + cosine similarity search
ranking.py             LightGBM ranking model
model_comparison.py    baselines + SHAP explainability
ab_testing.py          power analysis + simulated A/B test
cold_start.py          handling brand-new items with no history
generation.py          grounded LLM explanations (+ offline fallback)
pipeline.py            wires everything together
evaluate.py            retrieval precision@K
app.py                 Streamlit dashboard (5 tabs)
```

## License

MIT — do whatever you want with it.
