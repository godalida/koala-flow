# Example: Basic Churn Prediction (Local)
#
# This demonstrates running a "Churn Prediction" model end-to-end.
# It uses a mock data source (list of dicts) but writes to a real local DuckDB file.

import sys
import os
import pickle
from datetime import datetime

# Hack to import local package for development
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from koala_flow.core import InferencePipeline
from koala_flow.adapters import PickleAdapter
import dlt
import narwhals as nw
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# --- 1. Setup Mock Model ---
def create_dummy_model():
    print("Training dummy model...")
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = LogisticRegression()
    clf.fit(X, y)
    
    with open("churn_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Model saved to churn_model.pkl")

# --- 2. Setup Source Data (dlt) ---
def mock_crm_data():
    """Generates fake user data as a generator."""
    import random
    for i in range(1, 51):
        yield {
            "user_id": i,
            "feature_1": random.uniform(0, 1),
            "feature_2": random.uniform(0, 1),
            "feature_3": random.uniform(0, 1),
            "feature_4": random.uniform(0, 1),
            "signup_date": datetime.now().isoformat()
        }

# --- 3. Run Pipeline ---
if __name__ == "__main__":
    if not os.path.exists("churn_model.pkl"):
        create_dummy_model()

    # Create the pipeline instance
    pipeline = InferencePipeline(
        name="churn_predictor_v1",
        
        # Load our dummy model
        model=PickleAdapter("churn_model.pkl", version="1.0"),
        
        # Use local DuckDB for results
        destination=dlt.destinations.duckdb("analytics.db"),
        
        # Optional: Define feature transformation logic
        prep_fn=lambda df: df.select(["feature_1", "feature_2", "feature_3", "feature_4"])
    )

    # Execute!
    # dlt will pull from 'mock_crm_data', 'koala_flow' will batch process it, 
    # run inference, and 'dlt' will insert results into 'analytics.db'.
    pipeline.run(source=mock_crm_data(), table_name="scored_users")
    
    print("\n✅ Pipeline complete! Run this to verify results:")
    print("duckdb analytics.db \"SELECT * FROM ml_predictions.scored_users LIMIT 5\"")
