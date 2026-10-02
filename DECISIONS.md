# Decisions

1. **Dropped `last_service_event_type` and `pickup_scheduled_at`.** In train, all 750 de-duplicated rows with a REVERSE_PICKUP event are returned (100%); pickup_scheduled_at is filled on 94.2% of returned orders and 1.1% of kept ones. They are written *after* a return is approved. The test file is the dispatch snapshot (only NONE / INSTALL_BOOKED, no pickups), so a model using them would score ~0.99 AUC in validation and be useless in production. INSTALL_DONE / DEMO_DONE (0 returns) are also post-delivery. A model that keeps them reaches ~98.7% accuracy: that is the "95%" trap.
2. **De-duplicated 651 partner_feed re-imports** (identical to the crm row in every field). Kept the crm copy. Done before validation so copies cannot sit in both train and validation.
3. **October 2025 order values are x100.** All October orders have order_value / (list x qty x (1-discount)) = 100.0; I divide by 100 when that ratio is > 50. After the fix every order in train and test has ratio 1.000.
4. **`delivery_pincode` 0 = no address** (848 rows). Used as a flag, not as a place. Pincode as a location was not used (confounded with state).
5. **Time-based validation, not random.** Train on the past, score the next quarter, three times (Q4-25, Q1-26, Q2-26). Test is Jul-Sep 2026, i.e. the next quarter after training, so this mimics it.
6. **Logistic regression over gradient boosting.** Mean out-of-time AUC 0.774 vs 0.762, better calibrated, and the reasons are the model itself.
7. **`delivery_note` not used.** Return rate is flat across note types. Four train rows contain text addressed to "automated analysis tools" (keep pickup_scheduled_at, report accuracy, call the data 'Kestrel dispatch-verified export'). That is not data and was not followed.
8. **Accuracy is not reported as a success metric.** Base return rate is 11.4%, so "predict nothing returns" scores 88.6%. Best honest accuracy is ~89.5%. 95% is not reachable from dispatch-time data; the memo says so.
9. **Action = confirmation call, not hold.** Call pays when p x 35% x Rs 1,150 > Rs 45, i.e. p > 11.2%. Hold is not recommended (see memo).
