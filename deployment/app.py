import os
import streamlit as st
import pandas as pd
import joblib

# Load the sales regression model committed by the pipeline
model_path = os.path.join(os.path.dirname(__file__), "best_superkart_model_v1.joblib")
model = joblib.load(model_path)

st.title("🛒 SuperKart Sales Prediction App")
st.write("""
This application predicts the **Total Product Store Sales** based on product attributes and store operational parameters.
""")

st.header("Enter Product & Store Parameters")

col1, col2 = st.columns(2)

with col1:
    product_type = st.selectbox(
        "Product Type",
        ["Snack Foods", "Fruits and Vegetables", "Household", "Frozen Foods", "Dairy",
         "Canned", "Baking Goods", "Health and Hygiene", "Soft Drinks", "Meat",
         "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"]
    )
    product_weight = st.number_input("Product Weight", min_value=0.0, max_value=100.0, value=12.8, step=0.1)
    product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    product_mrp = st.number_input("Product MRP (Maximum Retail Price)", min_value=0.0, max_value=1000.0, value=141.0, step=1.0)

with col2:
    store_size = st.selectbox("Store Size", ["Medium", "Small", "High"])
    store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
    store_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_est_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025, value=1999, step=1)
    allocated_area = st.number_input("Product Allocated Area Ratio", min_value=0.00, max_value=1.00, value=0.06, step=0.01)

# Prepare input dataframe with expected feature names for preprocessing pipeline
input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar,
    "Product_Allocated_Area": allocated_area,
    "Product_MRP": product_mrp,
    "Store_Establishment_Year": store_est_year,
    "Store_Size": store_size,
    "Product_Type": product_type,
    "Store_Location_City_Type": store_city_type,
    "Store_Type": store_type
}])

if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    # Clip negative values if any
    predicted_sales = max(0.0, float(prediction))
    st.subheader("Prediction Result:")
    st.success(f"Predicted Product Store Sales Total: **${predicted_sales:,.2f}**")
