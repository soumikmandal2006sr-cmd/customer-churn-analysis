# Customer Churn Prediction & Retention Offer Engine

A machine learning project that predicts which subscription customers are likely to churn and recommends a specific retention offer for each one, based on *why* they are likely to leave.

> **Read this first:** the dataset is synthetic, and its labels follow hard rules, so the model scores are far higher than any real churn model would achieve. The retention-offer success rates are **assumptions**, not measured results. This project demonstrates a method. It does not demonstrate real-world lift. Details are in [Limitations](#limitations).

---

## What it does

1. **Predicts churn probability** for each customer with a LightGBM classifier.
2. **Calibrates** the probabilities so they can be used in an expected-value calculation.
3. **Explains** each prediction with SHAP and identifies each customer's main actionable churn driver.
4. **Recommends an offer** (payment plan, free account review, 20% discount, or no action) by maximizing expected value.
5. **Serves a dashboard** (Streamlit) to filter, rank and look up customers.

## Dataset

- Source: Kaggle "Customer Churn Dataset" (author: muhammadshahidazeem), training file, 440,832 usable rows (one fully empty row removed).
- Columns used: Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, Last Interaction. Target: Churn.
- The raw CSV is not included in this repository because of its size. Download it from Kaggle and place it in `data/`.

### Data decisions

| Decision | Reason |
|---|---|
| Excluded all **Monthly** contracts (87,104 rows) | Every one of them churned (100%). That is a hard rule in the data generator, not a behavior a model can learn from. 353,728 rows remain, with 46.1% churn. |
| Dropped **Gender** | Churn differs by gender (56.8% vs 38.4%, even after excluding Monthly), but using gender to decide who receives a retention offer is a fairness problem. |
| Dropped **CustomerID** | Identifier with no predictive meaning. |
| Did **not** build a revenue-per-month feature | Tenure ranges from 1 to 60, so dividing by it creates unstable values and adds no new information. |
| Stratified **60 / 20 / 20** split (train / validation / test) | Early stopping and calibration use the validation set. The test set is used only for final scores. |

### What the data looks like

The labels follow clear cutoffs. In the filtered data, churn is about 100% for customers with 6 or more support calls, more than 20 days of payment delay, total spend under about 500, or age over 50. Tenure and Usage Frequency carry almost no signal. This is why simple rules and tree models score so highly.

## Results

### Model comparison (validation set, before the final split)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.9314 | 0.9378 |
| Decision Tree (depth 4) | 0.9704 | 0.9683 |
| Random Forest | 0.9961 | 0.9964 |
| LightGBM | 0.9962 | 0.9964 |

LightGBM was chosen as the final model because it matches Random Forest and trains faster.

### Final model (held-out test set, 70,746 rows)

| Metric | Value |
|---|---|
| ROC-AUC | 0.9962 |
| PR-AUC | 0.9963 |
| Accuracy | 0.989 |
| Churn class precision / recall | 0.999 / 0.976 |
| False positives / missed churners | 19 / 779 |

The 779 missed churners all sit inside the "safe zone" (age 50 or under, 5 or fewer support calls, payment delay of 20 or less, spend of about 494 or more). They look like loyal customers and left anyway, which points to noise in the generator rather than a model weakness.

### Calibration

Brier score on the test set: 0.01045 raw, 0.01016 after isotonic calibration. The raw LightGBM probabilities were already well calibrated, so calibration made only a small difference. Predictions cluster near 0 and 1, so the calibration curve has few distinct points.

### Explainability

SHAP shows that Support Calls, Total Spend, Payment Delay, Age and Last Interaction drive predictions, in that order. Tenure, Usage Frequency, Subscription Type and Contract Length contribute almost nothing.

## Retention offer engine

For each customer and each offer:

```
EV(offer) = P(churn) x P(saved | offer, churn reason) x customer value - offer cost
```

The recommended offer is the one with the highest EV, or "no action" if none is positive.

- **Customer value:** Total Spend, used as a proxy for revenue at risk.
- **Churn reason:** the customer's strongest SHAP driver among the actionable features (Support Calls, Payment Delay, Total Spend, Last Interaction). Age is excluded because it cannot be acted on.
- **Offers and costs:** 20% discount (cost = 20% of value), free account review (cost 15), payment plan (cost 5).

### Assumed success rates (not measured)

| Churn reason | Discount | Account review | Payment plan |
|---|---|---|---|
| Payment | 0.25 | 0.10 | 0.40 |
| Service | 0.20 | 0.35 | 0.05 |
| Pricing | 0.35 | 0.10 | 0.05 |
| Engagement | 0.15 | 0.25 | 0.05 |

### Offer engine results (test set, under the assumed rates)

| Strategy | Total expected value |
|---|---|
| Matched offer per customer (engine) | 5,195,890 |
| Account review to all customers with p >= 0.5 | 3,894,660 |
| Payment plan to all customers with p >= 0.5 | 1,860,992 |
| Discount to all customers with p >= 0.5 | 498,313 |
| Discount to everyone | -5,254,462 |

- The engine scores about 33% higher than the best single-offer strategy. It targets almost the same customers as a simple p >= 0.5 rule (31,880 vs 31,859), so the gain comes from choosing the *right offer* for each reason, not from choosing different customers.
- Offer mix: no action 38,866, review 18,982, discount 8,394, plan 4,504.
- **Robustness:** if all assumed success rates are halved, 11.9% of recommendations change. If they are raised by 50%, 9.3% change.
- Expected values are in arbitrary units, not currency. Read the comparison between strategies, not the absolute numbers.

## Dashboard

```
streamlit run app.py
```

Filters by recommended offer, churn driver and minimum churn probability. Ranks by expected value, churn probability or spend. Includes a single-customer lookup. Reads `data/scored_customers.csv`.

## Project structure

```
churn_project/
  app.py                       Streamlit dashboard
  requirements.txt             minimal, for the hosted dashboard
  requirements-notebook.txt    full, for running the notebook
  data/
    scored_customers.csv       Scored test-set customers with recommended offers
  models/
    lgbm_churn.joblib
    isotonic_calibrator.joblib
    feature_columns.joblib
  notebooks/
    churn_project.ipynb        Full analysis, modelling and offer engine
```

## How to run

```
pip install -r requirements-notebook.txt
```

`requirements.txt` is the minimal file used by the hosted dashboard (streamlit and pandas only). Pinned versions matter here: `numpy==1.26.4` and `shap==0.46.0` were used together. Newer numpy versions broke other libraries in the environment this was built in.

To reproduce: place the raw CSV in `data/`, then run the notebook top to bottom. To use the dashboard only, run `pip install -r requirements.txt` and then `streamlit run app.py`.

## Limitations

- **Synthetic data with rule-based labels.** AUC near 0.996 reflects how easy the generator makes the problem. A real churn model would score much lower.
- **Offer success rates are assumed.** The dataset has no record of who received an offer or whether it worked. The engine's advantage over the baselines is calculated under the same assumptions it optimizes, so it is a consistency check, not evidence of real lift.
- **No uplift modelling.** True uplift needs treatment and control data. With real intervention data, the next step would be a two-model or causal uplift approach.
- **Total Spend is a rough value proxy.** It is cumulative spend, not forecast future revenue.
- **Most churners are deterministic.** Because predictions sit near 0 or 1, very few customers have uncertain churn probability, so the offer decision is driven mostly by the assumed success rates.
- **Last Interaction may be a leakage risk.** If it is recorded after a customer decides to leave, it would leak the outcome. This cannot be checked with the data provided.
- **SHAP drivers describe the model, not causes.** A customer's top SHAP feature explains why the model scored them high, not necessarily what would change their behavior.
- **Ranking by expected value favors high-spend customers.** The dashboard lets you rank by churn probability or spend instead.
- **Scoring covers the test set only.** The exported table is the 20% test split, not the whole customer base.

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, SHAP, matplotlib, seaborn, joblib, Streamlit.
