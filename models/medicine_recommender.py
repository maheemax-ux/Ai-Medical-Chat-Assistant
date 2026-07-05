

import pickle
import numpy as np
import pandas as pd
import joblib





class MedicineRecommender:
    def __init__(self, model_path: str = "/models/medicine_suggestion.pkl"):
        self.model_path = model_path
        self.model = None
        self._load()

    def _load(self):
        try:
            # Try joblib first (standard for scikit-learn models)
            self.model = joblib.load(self.model_path)
        except Exception:
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            except (FileNotFoundError, pickle.UnpicklingError, Exception):
                self.model = None

    @property
    def is_loaded(self) -> bool:
        # Medicine recommendations work via lookup even without a model
        return True

    def recommend(self, disease):
        """Return medicine suggestions for a given disease name."""
        disease_key = str(disease).strip()

        # Look up in the built-in disease-to-medicine mapping
        if disease_key in DISEASE_MEDICINES:
            return DISEASE_MEDICINES[disease_key]

        # Case-insensitive fallback
        for name, meds in DISEASE_MEDICINES.items():
            if name.lower() == disease_key.lower():
                return meds

        return []

    @staticmethod
    def _as_list(value):
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return [str(v) for v in value]
        return [str(value)]
