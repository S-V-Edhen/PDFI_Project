import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Punjab Farmer Distress Predictor", layout="wide")

st.title("Punjab Farmer Distress Prediction & Decision Support System")
st.write("Enter farmer details to predict PFDI and generate recommendations.")

# ---------------------------------------------------
# Safe imports
# ---------------------------------------------------
try:
    from src.app_utils import FEATURE_COLUMNS
    from src.decision_support import generate_recommendations
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

# ---------------------------------------------------
# Safe model loading
# ---------------------------------------------------
MODEL_PATH = Path("models") / "xgboost_pfdi.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.success("Model loaded successfully.")

# ---------------------------------------------------
# Input form
# ---------------------------------------------------
st.header("Farmer Profile Input")

with st.form("farmer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        land_owned = st.number_input("Land Owned", min_value=0.0, value=5.0)
        land_under_cultivation = st.number_input("Land Under Cultivation", min_value=0.0, value=4.0)
        land_lost_percent = st.number_input("Land Lost (%)", min_value=0.0, max_value=100.0, value=20.0)
        cause_land_loss = st.selectbox(
            "Cause of Land Loss",
            [0, 1, 2, 3],
            help="0=None, 1=Govt Acquisition, 2=Sale, 3=Inheritance"
        )
        land_type = st.selectbox(
            "Land Type",
            [0, 1, 2],
            help="0=Owned, 1=Leased, 2=Mixed"
        )

        annual_revenue = st.number_input("Annual Revenue", min_value=0.0, value=400000.0)
        annual_expenses = st.number_input("Annual Expenses", min_value=0.0, value=320000.0)
        net_profit = st.number_input("Net Profit", min_value=0.0, value=80000.0)

        profit_satisfaction = st.slider("Profit Satisfaction", 1, 5, 2)
        profit_trend = st.selectbox(
            "Profit Trend",
            [1, 2, 3],
            help="1=Declining, 2=Stable, 3=Increasing"
        )
        income_stability = st.slider("Income Stability", 1, 5, 2)

    with col2:
        seed_cost_burden = st.slider("Seed Cost Burden", 1, 5, 4)
        fertilizer_cost_burden = st.slider("Fertilizer Cost Burden", 1, 5, 4)
        pesticide_cost_burden = st.slider("Pesticide Cost Burden", 1, 5, 4)
        livestock_feed_cost_burden = st.slider("Livestock Feed Cost Burden", 1, 5, 3)

        msp_satisfaction = st.slider("MSP Satisfaction", 1, 5, 2)
        dairy_price_satisfaction = st.slider("Dairy Price Satisfaction", 1, 5, 2)
        price_fairness = st.slider("Price Fairness", 1, 5, 2)

        support_received_count = st.number_input(
            "Support Schemes Received Count",
            min_value=0,
            max_value=10,
            value=1
        )
        support_consistency = st.slider("Support Consistency", 1, 5, 2)
        compensation_satisfaction = st.slider("Compensation Satisfaction", 1, 5, 2)

        irrigation_source = st.selectbox(
            "Irrigation Source",
            [0, 1, 2, 3],
            help="0=Rain, 1=Canal, 2=Borewell, 3=Mixed"
        )
        water_availability = st.slider("Water Availability", 1, 5, 2)
        water_stability = st.slider("Water Stability", 1, 5, 2)

        labour_availability = st.slider("Labour Availability", 1, 5, 2)
        labour_cost_burden = st.slider("Labour Cost Burden", 1, 5, 4)

    with col3:
        livestock_owned = st.selectbox("Livestock Owned", [0, 1])
        livestock_operating_cost = st.slider("Livestock Operating Cost", 1, 5, 3)
        livestock_profitability = st.slider("Livestock Profitability", 1, 5, 2)
        vet_expenditure = st.slider("Veterinary Expenditure", 1, 5, 2)

        future_farming_confidence = st.slider("Future Farming Confidence", 1, 5, 2)
        children_interested = st.selectbox("Children Interested in Farming", [0, 1])
        education_level = st.slider("Education Level", 1, 5, 3)

        health_expenses = st.slider("Health Expenses", 1, 5, 2)
        marital_disputes = st.selectbox("Marital Disputes", [0, 1])
        chronic_illness = st.selectbox("Chronic Illness", [0, 1])
        education_expenses = st.slider("Education Expenses", 1, 5, 3)
        marriage_expenses = st.slider("Marriage Expenses", 1, 5, 2)
        unemployment = st.selectbox("Unemployment", [0, 1])
        alternative_income_source = st.selectbox("Alternative Income Source", [0, 1])

        pest_attack_frequency = st.slider("Pest Attack Frequency", 1, 5, 2)
        livestock_disease = st.selectbox("Livestock Disease", [0, 1])
        crop_failure_frequency = st.slider("Crop Failure Frequency", 1, 5, 2)
        lower_output_price = st.slider("Lower Output Price Pressure", 1, 5, 3)
        high_farm_expenses = st.slider("High Farm Expenses", 1, 5, 4)
        informal_debt = st.selectbox("Informal Debt", [0, 1])

    submitted = st.form_submit_button("Predict PFDI")

# ---------------------------------------------------
# Prediction block
# ---------------------------------------------------
if submitted:
    try:
        farmer_dict = {
            "land_owned": land_owned,
            "land_under_cultivation": land_under_cultivation,
            "land_lost_percent": land_lost_percent,
            "cause_land_loss": cause_land_loss,
            "land_type": land_type,
            "annual_revenue": annual_revenue,
            "annual_expenses": annual_expenses,
            "net_profit": net_profit,
            "profit_satisfaction": profit_satisfaction,
            "profit_trend": profit_trend,
            "income_stability": income_stability,
            "seed_cost_burden": seed_cost_burden,
            "fertilizer_cost_burden": fertilizer_cost_burden,
            "pesticide_cost_burden": pesticide_cost_burden,
            "livestock_feed_cost_burden": livestock_feed_cost_burden,
            "msp_satisfaction": msp_satisfaction,
            "dairy_price_satisfaction": dairy_price_satisfaction,
            "price_fairness": price_fairness,
            "support_received_count": support_received_count,
            "support_consistency": support_consistency,
            "compensation_satisfaction": compensation_satisfaction,
            "irrigation_source": irrigation_source,
            "water_availability": water_availability,
            "water_stability": water_stability,
            "labour_availability": labour_availability,
            "labour_cost_burden": labour_cost_burden,
            "livestock_owned": livestock_owned,
            "livestock_operating_cost": livestock_operating_cost,
            "livestock_profitability": livestock_profitability,
            "vet_expenditure": vet_expenditure,
            "future_farming_confidence": future_farming_confidence,
            "children_interested": children_interested,
            "education_level": education_level,
            "health_expenses": health_expenses,
            "marital_disputes": marital_disputes,
            "chronic_illness": chronic_illness,
            "education_expenses": education_expenses,
            "marriage_expenses": marriage_expenses,
            "unemployment": unemployment,
            "alternative_income_source": alternative_income_source,
            "pest_attack_frequency": pest_attack_frequency,
            "livestock_disease": livestock_disease,
            "crop_failure_frequency": crop_failure_frequency,
            "lower_output_price": lower_output_price,
            "high_farm_expenses": high_farm_expenses,
            "informal_debt": informal_debt,
        }

        # Validate feature coverage before prediction
        missing_features = [f for f in FEATURE_COLUMNS if f not in farmer_dict]
        extra_features = [f for f in farmer_dict if f not in FEATURE_COLUMNS]

        if missing_features:
            st.error(f"Missing features required by model: {missing_features}")
            st.stop()

        if extra_features:
            st.warning(f"Extra features not used by model: {extra_features}")

        farmer_df = pd.DataFrame([farmer_dict])[FEATURE_COLUMNS]

        pfdi = float(model.predict(farmer_df)[0])

        st.subheader("Predicted Punjab Farmer Distress Index (PFDI)")
        st.metric("PFDI", f"{pfdi:.4f}")

        if pfdi < 0.33:
            st.success("Distress Level: Low")
        elif pfdi < 0.66:
            st.warning("Distress Level: Moderate")
        else:
            st.error("Distress Level: High")

        st.subheader("Recommendations")
        tab1, tab2 = st.tabs(["Farmer Recommendations", "Policy Recommendations"])

        with tab1:
            try:
                baseline, farmer_recs = generate_recommendations(farmer_df, model, mode="farmer")
                st.write(f"Baseline PFDI: **{baseline}**")
                if farmer_recs.empty:
                    st.info("No farmer-level recommendations generated.")
                else:
                    st.dataframe(farmer_recs.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Farmer recommendation generation failed: {e}")

        with tab2:
            try:
                baseline, policy_recs = generate_recommendations(farmer_df, model, mode="policy")
                st.write(f"Baseline PFDI: **{baseline}**")
                if policy_recs.empty:
                    st.info("No policy-level recommendations generated.")
                else:
                    st.dataframe(policy_recs.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Policy recommendation generation failed: {e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")