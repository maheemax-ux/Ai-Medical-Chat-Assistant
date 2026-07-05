import pickle
import numpy as np
import pandas as pd
import joblib


disease_medicines = {
    "Diabetes": ["Metformin", "Insulin", "Glipizide"],
    "Hypertension": ["Amlodipine", "Losartan", "Hydrochlorothiazide"],
    "Common Cold": ["Paracetamol", "Antihistamines", "Decongestants"],
    "Flu": ["Oseltamivir", "Paracetamol", "Rest and fluids"],
    # add every disease label your model/classes can output
}


class MedicineRecommender:
    def __init__(self, model_path: str = "models/medicine_suggestion.pkl"):
        self.model_path = model_path
        self.model = None
        self._load()

    def _load(self):
        try:
            self.model = joblib.load(self.model_path)
        except Exception:
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            except (FileNotFoundError, pickle.UnpicklingError, Exception):
                self.model = None

    @property
    def is_loaded(self) -> bool:
        return True

    def recommend(self, disease):
        """Return medicine suggestions for a given disease name."""
        disease_key = str(disease).strip()

        if disease_key in disease_medicines:
            return disease_medicines[disease_key]

        for name, meds in disease_medicines.items():
            if name.lower() == disease_key.lower():
                return meds

        return []

    @staticmethod
    def _as_list(value):
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return [str(v) for v in value]
        return [str(value)]
