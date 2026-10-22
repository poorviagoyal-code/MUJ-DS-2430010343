# Final validation checklist

- [ ] Run the final notebook from a clean kernel or run the script end-to-end.
- [ ] Verify that the WHO target indicator is specifically the intended safely managed drinking-water service measure; `Dim1 == Total` alone may not prove the indicator category.
- [ ] Confirm the train/validation/test split is chronological and scalers are fit on training rows only.
- [ ] Ensure each model and the repeat-last-year baseline use identical test rows.
- [ ] Check predictions are within 0–100%; investigate and explain any out-of-range values rather than silently clipping them.
- [ ] Regenerate metrics, predictions, and figures from the same run.
- [ ] Ensure checkpoint and scaler parameters correspond to the final script.
- [ ] Update the presentation and report to reflect verified values only.
- [ ] Add the instructor as collaborator and confirm team-repository requirements with course staff.
