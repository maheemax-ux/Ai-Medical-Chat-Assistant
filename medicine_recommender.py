

import pickle
import numpy as np
import pandas as pd
import joblib


# Built-in disease-to-medicine mapping for the 41 standard diseases
DISEASE_MEDICINES = {
    "Fungal infection": ["Clotrimazole", "Fluconazole", "Terbinafine", "Ketoconazole"],
    "Allergy": ["Cetirizine", "Loratadine", "Fexofenadine", "Diphenhydramine"],
    "GERD": ["Omeprazole", "Pantoprazole", "Ranitidine", "Esomeprazole"],
    "Chronic cholestasis": ["Ursodeoxycholic acid", "Cholestyramine", "Rifampicin"],
    "Drug Reaction": ["Epinephrine", "Hydrocortisone", "Diphenhydramine", "Prednisolone"],
    "Peptic ulcer disease": ["Omeprazole", "Amoxicillin", "Clarithromycin", "Sucralfate"],
    "AIDS": ["Zidovudine", "Lamivudine", "Tenofovir", "Efavirenz"],
    "Diabetes": ["Metformin", "Glimepiride", "Insulin", "Sitagliptin"],
    "Gastroenteritis": ["ORS (Oral Rehydration Salts)", "Loperamide", "Ondansetron", "Zinc supplements"],
    "Bronchial Asthma": ["Salbutamol", "Budesonide", "Montelukast", "Ipratropium"],
    "Hypertension": ["Amlodipine", "Losartan", "Enalapril", "Hydrochlorothiazide"],
    "Migraine": ["Sumatriptan", "Ibuprofen", "Propranolol", "Topiramate"],
    "Cervical spondylosis": ["Ibuprofen", "Diclofenac", "Gabapentin", "Methocarbamol"],
    "Paralysis (brain hemorrhage)": ["Mannitol", "Nimodipine", "Physiotherapy", "Aspirin (post-recovery)"],
    "Jaundice": ["Ursodeoxycholic acid", "Vitamin K", "Rest & hydration", "Phenobarbital"],
    "Malaria": ["Chloroquine", "Artemether-Lumefantrine", "Quinine", "Doxycycline"],
    "Chicken pox": ["Acyclovir", "Calamine lotion", "Paracetamol", "Antihistamines"],
    "Dengue": ["Paracetamol", "ORS (Oral Rehydration Salts)", "IV fluids", "Platelet monitoring"],
    "Typhoid": ["Azithromycin", "Ciprofloxacin", "Ceftriaxone", "Amoxicillin"],
    "Hepatitis A": ["Rest & hydration", "Paracetamol (low-dose)", "Vitamin supplements", "Hepatitis A vaccine"],
    "Hepatitis B": ["Tenofovir", "Entecavir", "Peginterferon alfa-2a", "Lamivudine"],
    "Hepatitis C": ["Sofosbuvir", "Ledipasvir", "Daclatasvir", "Ribavirin"],
    "Hepatitis D": ["Peginterferon alfa-2a", "Tenofovir", "Entecavir", "Supportive care"],
    "Hepatitis E": ["Rest & hydration", "Ribavirin (severe cases)", "Supportive care", "Vitamin supplements"],
    "Alcoholic hepatitis": ["Prednisolone", "Pentoxifylline", "N-Acetylcysteine", "Nutritional support"],
    "Tuberculosis": ["Isoniazid", "Rifampicin", "Pyrazinamide", "Ethambutol"],
    "Common Cold": ["Paracetamol", "Pseudoephedrine", "Dextromethorphan", "Vitamin C"],
    "Pneumonia": ["Amoxicillin", "Azithromycin", "Levofloxacin", "Ceftriaxone"],
    "Dimorphic hemorrhoids (piles)": ["Lidocaine cream", "Hydrocortisone suppository", "Diosmin", "Fiber supplements"],
    "Heart attack": ["Aspirin", "Clopidogrel", "Atorvastatin", "Nitroglycerin"],
    "Varicose veins": ["Diosmin", "Compression stockings", "Horse chestnut extract", "Troxerutin"],
    "Hypothyroidism": ["Levothyroxine", "Liothyronine", "Selenium supplements", "Iodine supplements"],
    "Hyperthyroidism": ["Methimazole", "Propylthiouracil", "Propranolol", "Radioactive iodine"],
    "Hypoglycemia": ["Glucose tablets", "Glucagon injection", "Dextrose IV", "Dietary management"],
    "Osteoarthritis": ["Acetaminophen", "Ibuprofen", "Glucosamine", "Diclofenac gel"],
    "Arthritis": ["Methotrexate", "Ibuprofen", "Hydroxychloroquine", "Prednisolone"],
    "(vertigo) Paroxysmal Positional Vertigo": ["Meclizine", "Betahistine", "Dimenhydrinate", "Epley maneuver"],
    "Acne": ["Benzoyl peroxide", "Tretinoin", "Clindamycin gel", "Doxycycline"],
    "Urinary tract infection": ["Nitrofurantoin", "Trimethoprim", "Ciprofloxacin", "Fosfomycin"],
    "Psoriasis": ["Methotrexate", "Calcipotriol", "Betamethasone", "Cyclosporine"],
    "Impetigo": ["Mupirocin ointment", "Fusidic acid", "Cephalexin", "Amoxicillin-clavulanate"],
}


class MedicineRecommender:
    def __init__(self, model_path: str = "/Users/maheenakther/Downloads/mp1/models/medicine_suggestion.pkl"):
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
