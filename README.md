# Hybrid News Feed Recommender — Prototype

A personalized news feed recommendation engine that balances relevance with discovery — designed to avoid echo chambers while still surfacing content a user is likely to enjoy. Prototyped and validated on the MIND-small dataset before porting to a production feed of short, tweet-length news posts.

## Why this exists

Most recommendation engines optimize purely for engagement, which tends to trap users in an echo chamber of increasingly narrow content. This project explicitly architects *against* that by blending three recommendation strategies with different jobs:

- **Comfort** — show more of what the user already engages with
- **Social proof** — show what similar users are engaging with
- **Discovery** — deliberately surface rare, high-lift cross-topic connections the user wouldn't find on their own

## Architecture

```
                    User Actions
        (Likes, Dislikes, Comments, Reading Time,
              Shares, Clusters viewed)
                         │
                         ▼
              User Preference Profile
              (recency-weighted vector over
               unsupervised topic clusters)
                         │
        ┌────────────────┼──────────────────┐
        │                │                  │
        ▼                ▼                  ▼
  Item-Based CF     User-Based CF      Bridge Apriori
  (ALS, twin        (ALS, similar      (FP-Growth, low
   articles)         users)             support/high lift)
        │                │                  │
        └────────────────┼──────────────────┘
                         │
              Content-Based Similarity
           (embeddings, for cold-start /
              zero-history users)
                         │
                         ▼
              Score Combination Layer
                         │
              Freshness (exponential decay)
                         │
              Diversity Re-ranking (MMR)
                         │
                         ▼
              Final Personalized Feed
```

**Upstream, feeding the whole thing:** unsupervised topic discovery via BERTopic (embed → UMAP → HDBSCAN → c-TF-IDF labels), since the target deployment has no predefined news categories.

## Pipeline components

| Component | Method | Purpose |
|---|---|---|
| Topic discovery | BERTopic (sentence-transformer + UMAP + HDBSCAN) | Derives `cluster_id` per article — a substitute for hand-labeled categories |
| Item-based CF | Implicit ALS | "Twin articles" — recommend based on co-engagement patterns |
| User-based CF | Implicit ALS | Recommend what behaviorally similar users engaged with |
| Bridge discovery | FP-Growth (low support, lift > 2) | Mines rare, cross-cluster association rules for serendipity |
| Cold-start fallback | Sentence embedding cosine similarity | Content-based matching for new users/articles with no interaction history |
| Freshness | Exponential time decay | `score × exp(-λ × age_in_hours)` |
| Diversity re-ranking | MMR | Prevents near-duplicate articles from clustering together in one feed |

## Dataset

Prototyped on [MIND-small](https://www.kaggle.com/datasets/arashnic/mind-news-dataset) (Microsoft News Dataset). Only `Title`, `News ID`, and `Category` are used from article metadata — `Category` is held aside purely to validate cluster quality, never used as a training input, since the target application has no predefined categories.

**Note:** only the `MINDsmall_train` split was used; a held-out validation set was carved from the most recent 10% of impressions (by time) rather than relying on a separate dev split.

## Results (MIND-small validation split)

| Metric | This model | Popularity baseline |
|---|---|---|
| AUC | 0.570 | 0.523 |
| MRR | 0.299 | 0.251 |
| nDCG@5 | 0.278 | 0.223 |
| nDCG@10 | 0.333 | 0.290 |

The model outperforms a naive popularity baseline across all metrics, confirming the ALS + content-based blend adds real predictive signal beyond "just show what's popular." Absolute numbers sit below full deep-learning MIND benchmarks (NRMS/NAML/LSTUR, typically 0.65+ AUC), which is expected — those use neural encoders over title+abstract+entities, considerably more model capacity than this classical CF + association-rule approach.

**Clustering quality:** 633 unsupervised topic clusters, homogeneity score 0.50 (well above the ~0.05 floor for random assignment), mean per-cluster category purity 0.65 — indicating clusters are topically coherent despite having no supervision.

## What transfers to production, and what doesn't

| Artifact | Transfers as-is? |
|---|---|
| Sentence-transformer embedding model | ✅ Yes — general-purpose, not dataset-specific |
| BERTopic pipeline *methodology* (vectorizer config, UMAP/HDBSCAN params, outlier reduction) | ✅ Yes |
| Fitted UMAP/HDBSCAN clusterer | ❌ No — tuned to MIND's specific embedding distribution, must be refit on production data |
| ALS user/item factors | ❌ No — learned from MIND's specific users/articles |
| FP-Growth bridge rules | ❌ No — rules reference MIND's specific cluster IDs |

The validated piece is the **pipeline design**, not the trained weights — everything downstream of the embedding model gets refit once real production interaction data is available.

## Setup

```bash
pip install bertopic sentence-transformers implicit mlxtend umap-learn hdbscan scikit-learn
```

Run on Kaggle with the MIND-small dataset attached via "Add Data," or locally by pointing `NEWS_PATH`/`BEHAVIORS_PATH` at your own copy of `news.tsv`/`behaviors.tsv`.

## Roadmap

- [ ] Tune ALS hyperparameters (factors, regularization) and the ALS/content blend ratio
- [ ] Replace static 5:3:2 slotting with context-aware dynamic ratios (time of day, weekend, breaking news)
- [ ] Port pipeline to production schema (real interaction logs in place of MIND behaviors)
- [ ] Add comment sentiment analysis as a weighted signal
- [ ] Evaluate diversity metrics (ILD, coverage, novelty, serendipity@k) end-to-end, not just ranking metrics

## License

*(add your preferred license here)*
