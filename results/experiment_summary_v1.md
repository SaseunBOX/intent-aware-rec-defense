# Experiment Summary V1

## Current Project Stage
- Local prototype pipeline is runnable
- MIND-small preprocessing completed
- Aegis 2.0 integrated
- moral_education seed integrated
- external items injected into dev candidate set
- baseline and policy reranking experiments completed

## Main Data Status
- items.csv includes:
  - mind
  - aegis2
  - moral_education
- interactions_injected.csv has external harmful/safe candidates injected into dev sessions

## Key Results

### 1. Popularity Baseline
- eval_sessions: 73152
- HitRate@10: 0.614324
- NDCG@10: 0.309817

### 2. Injected Baseline Four-Metric Evaluation
- topk_rows: 698323
- eval_sessions: 38761
- HER: 0.044213
- OSR: 0.000000
- HitRate@10: 1.000000
- NDCG@10: 0.546064

### 3. Policy Rerank V1
- topk_rows: 698323
- eval_sessions: 38761
- HER: 0.041137
- OSR: 0.015501
- HitRate@10: 1.000000
- NDCG@10: 0.546064

### 4. Baseline vs Policy Delta
- HER reduction: 0.003076
- NDCG loss: 0.000000
- OSR numerator: 10346
- OSR denominator: 667448

## Current Interpretation
- Harmful exposure is reduced after policy reranking
- Utility metrics are currently unchanged
- OSR is now computed with a baseline-vs-policy comparison logic
- The prototype has entered a meaningful defense evaluation stage

## Next Planned Step
- Convert current results into a cleaner thesis-style comparison table
- Then move to a more formal experiment runner / table export
