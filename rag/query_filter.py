

import re

MEDICAL_KEYWORDS = [
    "disease", "symptom", "sympotm", "symtom", 
    "medicine", "medication", "drug", "treatment", "cure", "remedy",
    "diagnos", "doctor", "hospital", "clinic", "nurse", "patient",
    "fever", "pain", "ache", "sore", "infection", "virus", "bacteria",
    "cancer", "tumor", "diabetes", "blood pressure", "bp", "heart",
    "cardiac", "lung", "kidney", "liver", "stomach", "digest",
    "surgery", "vaccine", "allergy", "cough", "cold", "flu", "covid",
    "headache", "migraine", "nausea", "vomit", "diarrhea", "constipat",
    "rash", "itch", "wound", "injury", "bleed", "swelling", "swollen",
    "prescription", "dosage", "dose", "therapy", "health", "illness",
    "sick", "unwell", "pregnan", "mental health", "anxiety", "depress",
    "stress", "cholesterol", "asthma", "arthritis", "thyroid", "ulcer",
    "fatigue", "tired", "weak", "weakness", "dizzy", "dizziness",
    "fainting", "breathless", "breathing", "chest pain", "body ache",
    "body pain", "sleep", "insomnia", "appetite", "immune", "immunity",
    "wellbeing", "well-being", "recovery", "chronic", "acute",
    "dengue", "malaria", "typhoid", "chikungunya", "jaundice",
    "tuberculosis", "hepatit", "measles", "chickenpox", "pox",
    "pneumonia", "bronchitis", "sinus", "ulcer", "hernia", "stroke",
    "epilepsy", "seizure", "anemia", "anaemia", "obesity", "eczema",
    "psoriasis", "hiv", "aids", "std", "sti", "menstru", "period pain",
]

CLEARLY_OFF_TOPIC_PATTERNS = [
    r"\bstock (price|market)\b",
    r"\bwrite (a|me) (code|program|poem|song)\b",
    r"\bmovie|celebrity|football|cricket score\b",
    r"\brecipe for\b(?!.*(diet|nutrition|health))",
]


def keyword_prefilter(query: str):
    """
    Returns True (allow), False (block), or None (uncertain -> ask LLM).
    """
    q = query.lower()

    for pattern in CLEARLY_OFF_TOPIC_PATTERNS:
        if re.search(pattern, q):
            return False

    if any(kw in q for kw in MEDICAL_KEYWORDS):
        return True

    return None  # uncertain


def llm_classify_is_medical(query: str, llm_client) -> bool:
    """
    Ask the local LLM to classify the query. Used only when the keyword
    pre-filter is uncertain. If the LLM call itself fails (e.g. LM Studio
    unreachable), we fail OPEN (treat as medical) rather than silently
    blocking a possibly-legitimate health question just because the
    classifier call didn't work.
    """
    system_prompt = (
        "You are a strict binary classifier. Determine if the user's message "
        "is a medical, health, disease, symptom, or medication related "
        "question. Reply with exactly one word: YES or NO. Do not explain."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    reply = llm_client.chat(messages, temperature=0.0, max_tokens=5)

    if reply.strip().startswith("[LLM ERROR]") or reply.strip().startswith("[NO_LLM]"):
       
        return True

    return reply.strip().upper().startswith("Y")


def is_medical_query(query: str, llm_client=None) -> bool:
    result = keyword_prefilter(query)
    if result is not None:
        return result
    if llm_client is not None:
        return llm_classify_is_medical(query, llm_client)
   
    return True
