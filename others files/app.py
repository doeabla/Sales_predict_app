import streamlit as st
import pandas as pd
import pickle
import os
import random
from xgboost import XGBRFRegressor

# Function to load the ML component
def load_ml_comp(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)

# Main title and description
st.write("# ZANZIBAR STORES")
st.write("### PREDICT FUTURE SALES")

# Load ML model

folder_path = r"C:\Users\USER\Azubi LP4\Sales app"
file_name = 'pipeline.pkl'
ml_core_fp = os.path.join(folder_path, file_name)
ml_comp_dict = load_ml_comp(fp=ml_core_fp)

# User inputs
pred_date_from = st.date_input("Enter the starting date for your prediction")
pred_date_to = st.date_input("Enter the ending date for your prediction")
pred_on_promo = st.slider("Select Promo number", min_value=0, max_value=726)

# Days of the week
days_of_week = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
    "Friday": 5, "Saturday": 6, "Sunday": 7
}

# Checkbox to select all days
select_all_days = st.checkbox("Select all days of the week")

# Create a list to store selected day names
selected_day_names = []

if select_all_days:
    selected_day_names = list(days_of_week.keys())
else:
    # Multiselect for selecting specific days by name
    selected_day_names = st.multiselect("Select specific days", list(days_of_week.keys()))

# Map selected day names to numeric values
selected_days = [days_of_week[day_name] for day_name in selected_day_names]

# Display the selected days (day names) in the interface
st.write("Selected Days:", ", ".join(selected_day_names))

# Display the corresponding numeric values
st.write("Numeric Values:", selected_days)

# Clusters
cluster_list = list(range(1, 18))

select_all_clusters = st.checkbox("Select All Clusters")
if select_all_clusters:
    selected_clusters = cluster_list
else:
    selected_clusters = st.multiselect("Selected Cluster", cluster_list)

# Stores
store_list = list(range(1, 55))

select_all_stores = st.checkbox("Select All Stores")
if select_all_stores:
    selected_stores = store_list
else:
    selected_stores = st.multiselect("Selected Stores", store_list)

# Categories
categories = [
    'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS',
    'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS',
    'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
    'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES',
    'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE',
    'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
    'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY',
    'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES',
    'SEAFOOD'
]

select_all_categories = st.checkbox("Select All Categories")
if select_all_categories:
    selected_categories = categories
else:
    selected_categories = st.multiselect("Select Categories", categories)

# Generate random values (remove these in your actual code and replace with user inputs)
pred_saleslag1 = round(random.uniform(0, 9), 2)
pred_saleslag2 = round(random.uniform(0, 9), 2)
pred_roll_mean = round(random.uniform(0, 5), 2)
pred_roll_std = round(random.uniform(0, 4), 2)

# Button to trigger prediction
if st.button("Predict"):
    # Create a dataframe with user inputs (replace with actual data)
    data = {
        "store_nbr": selected_stores,
        "Product": [", ".join(selected_categories)],
        "onpromotion": [pred_on_promo],
        "cluster": selected_clusters,
        "day_of_week": selected_days,
        "sales_lag_1": [pred_saleslag1],
        "sales_lag_2": [pred_saleslag2],
        "rolling_mean": [pred_roll_mean],
        "rolling_std": [pred_roll_std]
    }
    input_df = pd.DataFrame(data)
    
    # Use the loaded machine learning model for predictions
    predictions = ml_comp_dict.predict(input_df)
    
    # Display the predictions
    st.write("Predictions:", predictions)
