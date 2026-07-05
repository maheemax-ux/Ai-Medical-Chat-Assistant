
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from config import MEDICAL_DISCLAIMER
from models.disease_predictor import DiseasePredictor
from models.medicine_recommender import MedicineRecommender
from utils.hospital_finder import geocode_address, find_nearby_hospitals, get_user_location_by_ip

st.set_page_config(page_title="AI Medical Chat Assistant", page_icon="🩺", layout="wide")

@st.cache_resource
def get_disease_predictor():
    return DiseasePredictor()


@st.cache_resource
def get_medicine_recommender():
    return MedicineRecommender()


@st.cache_resource
def get_rag_engine():
    from rag.rag_engine import RAGEngine
    return RAGEngine()


disease_predictor = get_disease_predictor()
medicine_recommender = get_medicine_recommender()

st.title("🩺 Medical Assistant")
st.caption(MEDICAL_DISCLAIMER)

tab1, tab2, tab3 = st.tabs(
    ["🔬 Disease & Medicine", "🏥 Nearby Hospitals", "💬 Medical Chatbot"]
)

with tab1:
    st.subheader("Symptom-based Disease Prediction")

    if not disease_predictor.is_loaded:
        st.error(
            "disease_model.pkl could not be loaded. Check the path in "
            "config.py (DISEASE_MODEL_PATH) and make sure the file exists."
        )
    else:
        known_symptoms = disease_predictor.get_known_symptoms()

        col_a, col_b = st.columns(2)
        with col_a:
            selected_symptoms = []
            if known_symptoms:
                selected_symptoms = st.multiselect(
                    "Select known symptoms",
                    options=known_symptoms,
                    help="These options come from the model's known feature list.",
                )
            else:
                st.info(
                    "No fixed symptom list detected on the model. Enter symptoms "
                    "manually below (comma-separated)."
                )
        with col_b:
            manual_symptoms_text = st.text_input(
                "Or enter additional symptoms (comma-separated)",
                placeholder="e.g. fever, headache, fatigue",
            )

        manual_symptoms = [s.strip() for s in manual_symptoms_text.split(",") if s.strip()]
        all_symptoms = list(dict.fromkeys(selected_symptoms + manual_symptoms))

        top_n = st.slider("How many top disease matches to show?", 1, 10, 5)

        if st.button("🔍 Predict Disease(s)", type="primary"):
            if not all_symptoms:
                st.warning("Please select or enter at least one symptom.")
            else:
                results = disease_predictor.predict(all_symptoms, top_n=top_n)
                if not results:
                    st.error(
                        "Prediction failed. This likely means the model's "
                        "expected input format differs from the assumption "
                        "in models/disease_predictor.py — run "
                        "`python inspect_models.py` and adjust `_vectorize()`."
                    )
                else:
                    st.session_state["predicted_diseases"] = results
                    st.success("Prediction complete.")
                    for disease, conf in results:
                        st.progress(min(max(float(conf), 0.0), 1.0), text=f"{disease} — {conf*100:.1f}%")

        st.divider()
        st.subheader("Medicine Suggestion")

        disease_options = []
        if "predicted_diseases" in st.session_state:
            disease_options = [d for d, _ in st.session_state["predicted_diseases"]]
        all_known_diseases = disease_predictor.get_known_diseases()

        chosen_diseases = st.multiselect(
            "Select disease(s) to get medicine suggestions for",
            options=list(dict.fromkeys(disease_options + all_known_diseases)),
            default=disease_options,
        )

        if st.button("💊 Suggest Medicines"):
            if not medicine_recommender.is_loaded:
                st.error(
                    "madicine_suggestion.pkl could not be loaded. Check "
                    "MEDICINE_MODEL_PATH in config.py."
                )
            elif not chosen_diseases:
                st.warning("Select at least one disease first.")
            else:
                for disease in chosen_diseases:
                    meds = medicine_recommender.recommend(disease)
                    st.markdown(f"**{disease}**")
                    if meds:
                        for m in meds:
                            st.markdown(f"- {m}")
                    else:
                        st.markdown("_No medicine suggestion found for this disease._")
                st.info(MEDICAL_DISCLAIMER)

with tab2:
    st.subheader("Find Nearby Hospitals")
    st.caption("Powered by OpenStreetMap (Nominatim + Overpass) — no API key required.")

    # Initialize session state
    if "hospital_lat" not in st.session_state:
        st.session_state.hospital_lat = None
        st.session_state.hospital_lon = None
        st.session_state.hospital_results = None
        st.session_state.hospital_loc_name = None

    # Browser GPS geolocation (triggers "Allow Location" popup)
    from streamlit_js_eval import get_geolocation
    location = get_geolocation()

    if location and st.session_state.hospital_lat is None:
        coords = location.get("coords", {})
        if coords.get("latitude") and coords.get("longitude"):
            st.session_state.hospital_lat = coords["latitude"]
            st.session_state.hospital_lon = coords["longitude"]
            st.session_state.hospital_loc_name = f"Your GPS Location"
            st.session_state.hospital_results = None

    # Fallback to IP if GPS not available yet
    if st.session_state.hospital_lat is None:
        result = get_user_location_by_ip()
        if result:
            st.session_state.hospital_lat, st.session_state.hospital_lon, st.session_state.hospital_loc_name = result

    # Show detected location
    if st.session_state.hospital_lat is not None:
        st.success(f"📍 {st.session_state.hospital_loc_name} ({st.session_state.hospital_lat:.4f}, {st.session_state.hospital_lon:.4f})")

    # Option to change location
    with st.expander("📌 Change location", expanded=(st.session_state.hospital_lat is None)):
        loc_mode = st.radio("Search by:",
                             ["Enter address / city", "Enter latitude & longitude"], horizontal=True)

        if loc_mode == "Enter address / city":
            address = st.text_input("Enter your address, city, or area", placeholder="e.g. Madurai, Tamil Nadu")
            if st.button("📍 Update Location"):
                if not address.strip():
                    st.warning("Please enter an address.")
                else:
                    with st.spinner("Looking up address..."):
                        coords = geocode_address(address)
                    if not coords:
                        st.error("Couldn't find that location. Try a more specific address.")
                    else:
                        st.session_state.hospital_lat, st.session_state.hospital_lon = coords
                        st.session_state.hospital_loc_name = address
                        st.session_state.hospital_results = None
                        st.rerun()
        else:
            c1, c2 = st.columns(2)
            with c1:
                lat_input = st.number_input("Latitude", value=0.0, format="%.6f")
            with c2:
                lon_input = st.number_input("Longitude", value=0.0, format="%.6f")
            if st.button("📍 Update Location", key="update_latlon"):
                st.session_state.hospital_lat = lat_input
                st.session_state.hospital_lon = lon_input
                st.session_state.hospital_loc_name = f"{lat_input:.4f}, {lon_input:.4f}"
                st.session_state.hospital_results = None
                st.rerun()

    radius_km = st.slider("Search radius (km)", 1, 20, 5)

    # Search and display hospitals
    if st.session_state.hospital_lat is not None and st.session_state.hospital_lon is not None:
        lat = st.session_state.hospital_lat
        lon = st.session_state.hospital_lon

        # Search for hospitals if not already done
        if st.session_state.hospital_results is None:
            with st.spinner("Searching for nearby hospitals..."):
                st.session_state.hospital_results = find_nearby_hospitals(lat, lon, radius_m=int(radius_km * 1000))

        hospitals = st.session_state.hospital_results

        if not hospitals:
            st.info("No hospitals found in this radius. Try increasing the search radius.")
        else:
            st.success(f"Found {len(hospitals)} hospital(s) nearby.")
            map_points = [{"lat": h["lat"], "lon": h["lon"]} for h in hospitals] + [{"lat": lat, "lon": lon}]
            st.map(map_points, latitude="lat", longitude="lon")

            for h in hospitals:
                with st.container(border=True):
                    st.markdown(f"**🏥 {h['name']}**")
                    st.markdown(f"📍 {h['address']}")
                    st.markdown(f"📞 {h['phone']}")
                    st.markdown(
                        f"[Open in Google Maps](https://www.google.com/maps/search/?api=1&query={h['lat']},{h['lon']})"
                    )
    else:
        st.info("📍 Please allow location access in your browser, or use the options above to enter your location.")

with tab3:
    st.subheader("Medical Chatbot")
    st.caption(
        "Ask health/medical questions. This chatbot uses Retrieval-Augmented "
        "Generation (RAG) over a medical knowledge base, PubMed, and "
        "MedlinePlus, and will politely decline non-medical questions."
    )

    rag_engine = get_rag_engine()

    use_web = st.checkbox(
        "🌐 Include live internet lookup (PubMed + MedlinePlus)",
        value=True,
        help="Supplements your local knowledge base with fresh results from "
             "PubMed and MedlinePlus for each question.",
    )

    if not rag_engine.llm.is_configured:
        st.info(
            "ℹ️ No AI model is configured yet, so answers will show the raw "
            "retrieved information instead of a generated summary. To enable "
            "AI-written answers, set `LLM_PROVIDER` to `\"openai\"`, "
            "`\"anthropic\"`, or `\"gemini\"` in `config.py` and add the "
            "matching API key."
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role", "content"}

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a medical question...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = rag_engine.answer(
                    user_query,
                    chat_history=st.session_state.chat_history[:-1],
                    use_web=use_web,
                )
            st.markdown(result["answer"])
            if result["sources"]:
                st.caption("Sources: " + ", ".join(sorted(set(result["sources"]))))

        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
