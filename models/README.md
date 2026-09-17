# models/

This directory is reserved for saved model files (e.g., `.pkl`, `.joblib`).

Trained models are not committed to Git to keep the repository lightweight
and to avoid version-control issues with binary files.

## Re-training

All models are re-trained from scratch each time the Streamlit app runs
(results are cached in the Streamlit session). This ensures reproducibility:
the same dataset + the same `random_state=42` always produces identical results.

## If you want to save models

You can extend `src/segmentation.py` and `src/campaign_model.py` with:

```python
import joblib
joblib.dump(km_model, "models/kmeans.pkl")
joblib.dump(lr_model, "models/logistic_regression.pkl")
```

And load with:

```python
km_model = joblib.load("models/kmeans.pkl")
```

Add `joblib` to `requirements.txt` if you do this.
