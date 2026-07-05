

import pickle
import numpy as np
import joblib


# Label-to-disease mapping for the RandomForestClassifier
# (trained on the standard 41-disease symptom dataset with LabelEncoder)
DISEASE_LABELS = {
    0: "Fungal infection",
    1: "Allergy",
    2: "GERD",
    3: "Chronic cholestasis",
    4: "Drug Reaction",
    5: "Peptic ulcer disease",
    6: "AIDS",
    7: "Diabetes",
    8: "Gastroenteritis",
    9: "Bronchial Asthma",
    10: "Hypertension",
    11: "Migraine",
    12: "Cervical spondylosis",
    13: "Paralysis (brain hemorrhage)",
    14: "Jaundice",
    15: "Malaria",
    16: "Chicken pox",
    17: "Dengue",
    18: "Typhoid",
    19: "Hepatitis A",
    20: "Hepatitis B",
    21: "Hepatitis C",
    22: "Hepatitis D",
    23: "Hepatitis E",
    24: "Alcoholic hepatitis",
    25: "Tuberculosis",
    26: "Common Cold",
    27: "Pneumonia",
    28: "Dimorphic hemorrhoids (piles)",
    29: "Heart attack",
    30: "Varicose veins",
    31: "Hypothyroidism",
    32: "Hyperthyroidism",
    33: "Hypoglycemia",
    34: "Osteoarthritis",
    35: "Arthritis",
    36: "(vertigo) Paroxysmal Positional Vertigo",
    37: "Acne",
    38: "Urinary tract infection",
    39: "Psoriasis",
    40: "Impetigo",
}


class DiseasePredictor:
    def __init__(self, model_path: str = "/models/RANmodel.pkl"):
        self.model_path = model_path
        self.model = None
        self.symptom_list = None  
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

    def _label_to_name(self, label):
        """Convert a numeric label to a disease name."""
        key = int(label)
        return DISEASE_LABELS.get(key, f"Unknown disease ({key})")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def get_known_diseases(self):
        if self.model is not None and hasattr(self.model, "classes_"):
            return [self._label_to_name(c) for c in self.model.classes_]
        return []

    def get_known_symptoms(self):

        if self.symptom_list:
            return self.symptom_list
        if self.model is not None and hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        return []

    def _vectorize(self, symptoms: list):

        known = self.get_known_symptoms()
        if not known:
            raise ValueError(
                "Could not determine the model's expected feature/symptom "
                "list. Please set self.symptom_list in disease_predictor.py "
                "(load it from wherever you saved it during training)."
            )
        vec = np.zeros(len(known), dtype=int)
        symptom_set = {s.strip().lower() for s in symptoms}
        for i, name in enumerate(known):
            if name.strip().lower() in symptom_set:
                vec[i] = 1
        return vec.reshape(1, -1)

    def predict(self, symptoms: list, top_n: int = 5):

        if not self.is_loaded:
            return []

        try:
            X = self._vectorize(symptoms)
        except ValueError:
            return []

        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                classes = self.model.classes_
                ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
                # Convert numeric labels to disease names
                return [(self._label_to_name(cls), prob) for cls, prob in ranked[:top_n]]
            else:
                # Fallback: model only supports predict(), no confidence score
                pred = self.model.predict(X)[0]
                return [(self._label_to_name(pred), 1.0)]
        except Exception as e:
            print(f"[DiseasePredictor] prediction failed: {e}")
            return []
