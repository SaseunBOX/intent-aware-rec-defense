# Chapter 4 and Chapter 5 Writing Summary V1

## 1. Model Quality Summary
- intent_model_v1: overall accuracy = 0.647296 (Session-level intent probability model)
- risk_model_v1: overall accuracy = 0.995675 (Item-level risk probability model)

## 2. Main Experimental Findings
- The injected baseline produced HER = 0.044213, OSR = 0.000000, HitRate@10 = 1.000000, and NDCG@10 = 0.546064.
- The thesis-style Full Pipeline V2 reduced HER to 0.000048, achieving a HER reduction of 0.044165 relative to the injected baseline.
- Under Full Pipeline V2, OSR = 0.269293, NDCG@10 = 0.532155, and utility loss vs baseline = 0.025470.
- Compared with the injected baseline, the full pipeline substantially reduced harmful exposure while preserving a high recommendation utility level.

## 3. Ablation Interpretation
- The risk-only control reduced HER more strongly than the intent-only control in the current setup.
- The intent-only control changed OSR more noticeably than the risk-only control, indicating that user-intent-sensitive routing mainly affected acceptable-content retention.
- The earlier Full_Policy_V1 provided a useful bridge from simple rule-based reranking to the more thesis-aligned Full Pipeline V2.

## 4. Operating Points
- Balanced mode selected: Balanced_Mode_Selected with HER = 0.000000, OSR = 0.269294, NDCG@10 = 0.532155.
- High-safety mode selected: High_Safety_Mode_Selected with HER = 0.000000, OSR = 0.269294, NDCG@10 = 0.532155.
- In the current sweep, the two operating points coincided, suggesting that the tested search range already approached a dominant safety-preferred region of the trade-off frontier.

## 5. Suggested Chapter 5 Conclusion Points
- The proposed intent-aware multi-stage defense can sharply suppress harmful exposure in offline recommendation simulation.
- The strongest gains come from combining probabilistic intent estimation, probabilistic risk estimation, score fusion, and hard intervention actions.
- The current prototype demonstrates feasibility, but future work should expand the safe replacement pool, refine classifier calibration, and broaden the sweep range for more clearly separated operating modes.
