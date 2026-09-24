---

### Recommendation to the client *(draft — to be validated by the team)*

**Deploy XGBoost in production, with two safeguards. Keep the logistic regression as a challenger. Do not deploy TabPFN.**

The figures below are for LGD = 60%, on the 20,000 test loans.

| | Evidence |
|---|---|
| **Why XGBoost** | Profit is **+€0.7M** higher than the logistic regression (paired bootstrap 95% CI: +€0.37M to +€1.04M), i.e. about +4% or **≈ €35 per applicant**. AUC is 0.916 vs 0.906. Every decision can still be explained **exactly** with TreeSHAP, in milliseconds, which meets the GDPR art. 22 / EU AI Act requirement. |
| **Why not TabPFN** | Same AUC and profit as XGBoost (+€0.03M, 95% CI −€0.25M to +€0.29M: no gain). It costs about 1,000× more per decision, its explanations are only approximations, and its decisions depend on which rows are in its context. It remains a useful **benchmark** of what "state of the art without tuning" achieves. |
| **What the logistic regression buys** | The most **transparent** (exact coefficients) and the most **stable** model: 0.2% of decisions flip under 1% input noise, vs 2.2–2.3% for XGBoost and TabPFN. Its cost is the price of full transparency: ≈ €35 of profit per applicant. |
| **Fairness** | Identical across the three models. None uses gender or marital status. The gender gap is **certified** below 2 points (TOST). Two points are shared by all models and must be owned by the lender: education gaps, explained by employment; and a lower approval rate for **divorced good payers**, through proxies. Near-total exclusion of the unemployed and of students is a **policy choice**, not a statistical artifact. |

**Safeguards for the XGBoost deployment:**
1. **Grey-zone review.** Send applicants scored within ±0.03 of the threshold to a human analyst. This is where input noise flips decisions, mostly because of the stepped debt-to-income pattern.
2. **Challenger monitoring.** Score every application with the logistic regression as well. Any disagreement between the two models is logged and reviewed. This monitors drift and gives a second, fully transparent explanation.

*If the client values full transparency and stability above ≈ €35 per applicant, the logistic regression is the defensible alternative.*
