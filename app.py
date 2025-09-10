import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Error handling for model loading
try:
    model = joblib.load('best_model.pkl')
except AttributeError as e:
    st.error("Model loading failed due to version incompatibility: {}. This is likely due to a mismatch between the scikit-learn version used to create 'best_model.pkl' (e.g., 1.5.1) and the current version (1.7.1). Please either:\n- Install scikit-learn==1.5.1 using 'pip install scikit-learn==1.5.1' and rerun.\n- Regenerate the model with the current scikit-learn version by retraining and saving it again.\nSee the project documentation for instructions.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}. Ensure 'best_model.pkl' exists and all dependencies (scikit-learn, pandas, joblib, etc.) are installed.")
    st.stop()

# App title
st.title('YouTube Ad Revenue Predictor')

# User inputs
views = st.number_input('Views', min_value=0.0)
likes = st.number_input('Likes', min_value=0.0)
comments = st.number_input('Comments', min_value=0.0)
watch_time_minutes = st.number_input('Watch Time (minutes)', min_value=0.0)
video_length_minutes = st.number_input('Video Length (minutes)', min_value=0.0)
subscribers = st.number_input('Subscribers', min_value=0.0)
category = st.selectbox('Category', ['Gaming', 'Education', 'Entertainment', 'Tech', 'Lifestyle'])
device = st.selectbox('Device', ['Mobile', 'Desktop', 'Tablet'])
country = st.selectbox('Country', ['USA', 'India', 'UK', 'Brazil', 'Canada'])
year = st.number_input('Year', min_value=2020, max_value=2025)
month = st.number_input('Month', min_value=1, max_value=12)
day = st.number_input('Day', min_value=1, max_value=31)

# Feature engineering
engagement_rate = (likes + comments) / views if views > 0 else 0

# Prepare input
input_data = pd.DataFrame({
    'views': [views],
    'likes': [likes],
    'comments': [comments],
    'watch_time_minutes': [watch_time_minutes],
    'video_length_minutes': [video_length_minutes],
    'subscribers': [subscribers],
    'category': [category],
    'device': [device],
    'country': [country],
    'engagement_rate': [engagement_rate],
    'year': [year],
    'month': [month],
    'day': [day]
})

# Predict
if st.button('Predict Revenue'):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f'Predicted Ad Revenue: ${prediction:.2f}')
    except Exception as e:
        st.error(f"Prediction failed: {e}. Please check input values or regenerate the model.")
        st.stop()

# Visualizations
st.header('Model Insights')
# Example: Feature Importance (placeholder, replace with actual importance if available)
fi_data = pd.DataFrame({
    'Feature': ['watch_time_minutes', 'views', 'engagement_rate', 'subscribers'],
    'Importance': [0.01, 0.001, 0.0005, 0.00001]  # Sample data, replace with real values
})
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=fi_data, ax=ax)
st.pyplot(fig)

# Basic Analytics
st.header('Dataset Sample Visuals')
# Load sample data (replace with actual data loading if available)
try:
    sample_data = pd.read_csv('youtube_data.csv')[:1000]  # Assuming sample data is saved
    fig2, ax2 = plt.subplots()
    sns.histplot(sample_data['ad_revenue_usd'], kde=True, ax=ax2)
    st.pyplot(fig2)
except FileNotFoundError:
    st.warning("Sample data 'youtube_data.csv' not found. Please ensure the dataset is available.")
except Exception as e:
    st.error(f"Error loading sample data: {e}")