import pandas as pd

variable_control = {
    "land_owned": "none",
    "land_under_cultivation": "both",
    "land_lost_percent": "none",
    "cause_land_loss": "none",
    "land_type": "none",

    "annual_revenue": "both",
    "annual_expenses": "both",
    "net_profit": "both",
    "profit_satisfaction": "farmer",
    "profit_trend": "both",
    "income_stability": "farmer",

    "seed_cost_burden": "both",
    "fertilizer_cost_burden": "both",
    "pesticide_cost_burden": "both",
    "livestock_feed_cost_burden": "both",

    "msp_satisfaction": "policy",
    "dairy_price_satisfaction": "both",
    "price_fairness": "policy",

    "support_received_count": "farmer",
    "support_consistency": "policy",
    "compensation_satisfaction": "policy",

    "water_availability": "both",
    "water_stability": "both",

    "labour_availability": "policy",
    "labour_cost_burden": "both",

    "livestock_operating_cost": "farmer",
    "livestock_profitability": "farmer",

    "future_farming_confidence": "both",
    "children_interested": "none",
    "alternative_income_source": "farmer",

    "crop_failure_frequency": "both",
    "lower_output_price": "policy",
    "high_farm_expenses": "both",
    "informal_debt": "both"
}

positive_features = [
    "profit_satisfaction",
    "income_stability",
    "msp_satisfaction",
    "price_fairness",
    "dairy_price_satisfaction",
    "support_received_count",
    "support_consistency",
    "compensation_satisfaction",
    "water_availability",
    "water_stability",
    "labour_availability",
    "livestock_profitability",
    "future_farming_confidence",
    "alternative_income_source"
]

negative_features = [
    "seed_cost_burden",
    "fertilizer_cost_burden",
    "pesticide_cost_burden",
    "livestock_feed_cost_burden",
    "labour_cost_burden",
    "livestock_operating_cost",
    "high_farm_expenses",
    "crop_failure_frequency",
    "informal_debt",
    "annual_expenses",
]

def improve_feature(feature, value):
    if feature in positive_features:
        if value in [0, 1]:
            return 1
        if value <= 5:
            return min(value + 1, 5)
        return value * 1.10

    if feature in negative_features:
        if value in [0, 1]:
            return 0
        if value <= 5:
            return max(value - 1, 1)
        return value * 0.90

    if feature == "net_profit":
        return value * 1.10

    if feature == "annual_revenue":
        return value * 1.10

    return value

def generate_recommendations(farmer_df, model, mode="farmer"):
    baseline_pfdi = float(model.predict(farmer_df)[0])
    rows = []

    for feature in farmer_df.columns:
        controller = variable_control.get(feature, "none")
        if controller not in [mode, "both"]:
            continue

        original = farmer_df[feature].iloc[0]
        improved = improve_feature(feature, original)

        if improved == original:
            continue

        modified = farmer_df.copy()
        modified.at[0, feature] = improved

        new_pfdi = float(model.predict(modified)[0])
        reduction = baseline_pfdi - new_pfdi

        rows.append({
            "Feature": feature,
            "Original": original,
            "Improved": improved,
            "New_PFDI": round(new_pfdi, 4),
            "PFDI_Reduction": round(reduction, 4)
        })

    recs = pd.DataFrame(rows)
    if not recs.empty:
        recs = recs.sort_values("PFDI_Reduction", ascending=False)

    return round(baseline_pfdi, 4), recs