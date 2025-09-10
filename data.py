import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Apply YouTube-inspired custom CSS
st.markdown(
    """
    <style>
    /* Main content area with YouTube white background */
    .main {
        background-color: #FFFFFF;
        color: #000000;
    }
    /* Sidebar with YouTube red background */
    .sidebar .sidebar-content {
        background-color: #FF0000;  /* YouTube red */
        color: white;
        border-right: 1px solid #cc0000;
    }
    /* Header with YouTube red */
    .css-1d391kg {
        background-color: #FF0000;
        color: white;
        padding: 10px 20px;
        border-bottom: 2px solid #cc0000;
    }
    .css-1d391kg h1 {
        color: white;
        text-align: left;
        font-size: 28px;
        margin: 0;
    }
    .css-1d391kg h3 {
        color: white;
        text-align: left;
        font-size: 18px;
        margin: 5px 0 0;
    }
    /* Buttons styled like YouTube */
    .stButton>button {
        background-color: #FF0000;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 2px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #cc0000;
        color: white;
    }
    /* Selectbox with YouTube style */
    .stSelectbox div {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #e0e0e0;
        border-radius: 2px;
    }
    /* Dataframe styling */
    .stDataFrame {
        background-color: #FFFFFF;
        border: 1px solid #e0e0e0;
        border-radius: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# YouTube-style header with logo placeholder
st.markdown(
    """
    <div style='background-color: #FF0000; padding: 10px 20px;'>
        <h1 style='color: white; text-align: left; font-size: 28px; margin: 0;'>CONTENT MONETIZATION MODELER</h1>
        <h3 style='color: white; text-align: left; font-size: 18px; margin: 5px 0 0;'>YouTube Ad Revenue Prediction</h3>
       
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation
page = st.sidebar.selectbox("Navigate", ["Dataset Upload", "EDA", "Preprocessing", "Data Visualization", "Model Training", "Prediction"])

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# Page 1: Dataset Upload
if page == "Dataset Upload":
    st.header("Dataset Upload")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
        st.dataframe(st.session_state.data.head())
    else:
        st.info("Please upload a CSV file to proceed. Alternatively, generate synthetic data.")
        if st.button("Generate Synthetic Dataset"):
            # Generate synthetic data as per project specs
            np.random.seed(42)
            num_rows = 122000
            video_ids = np.random.choice(range(10000), num_rows, replace=True)
            dates = pd.date_range(start='2020-01-01', periods=num_rows//100, freq='D').repeat(100)[:num_rows]
            views = np.random.poisson(lam=10000, size=num_rows).clip(1)
            likes = (views * np.random.uniform(0.01, 0.1, num_rows)).astype(int)
            comments = (views * np.random.uniform(0.001, 0.01, num_rows)).astype(int)
            watch_time_minutes = views * np.random.uniform(1, 5, num_rows)
            video_length_minutes = np.random.uniform(5, 30, num_rows)
            subscribers = np.random.poisson(lam=100000, size=num_rows)
            categories = np.random.choice(['Gaming', 'Education', 'Entertainment', 'Tech', 'Lifestyle'], num_rows)
            devices = np.random.choice(['Mobile', 'Desktop', 'Tablet'], num_rows)
            countries = np.random.choice(['USA', 'India', 'UK', 'Brazil', 'Canada'], num_rows)
            ad_revenue_usd = (views * 0.001 + watch_time_minutes * 0.01 + likes * 0.0005 + comments * 0.0002 + subscribers * 0.00001) * np.random.uniform(0.8, 1.2, num_rows)

            data = pd.DataFrame({
                'video_id': video_ids,
                'date': dates,
                'views': views,
                'likes': likes,
                'comments': comments,
                'watch_time_minutes': watch_time_minutes,
                'video_length_minutes': video_length_minutes,
                'subscribers': subscribers,
                'category': categories,
                'device': devices,
                'country': countries,
                'ad_revenue_usd': ad_revenue_usd
            })

            # Introduce ~5% missing and ~2% duplicates
            for col in ['views', 'likes', 'comments', 'watch_time_minutes']:
                data.loc[data.sample(frac=0.05).index, col] = np.nan
            duplicates = data.sample(frac=0.02)
            data = pd.concat([data, duplicates], ignore_index=True)
            data.to_csv('youtube_data.csv', index=False)

            st.session_state.data = data
            st.success("Synthetic dataset generated and loaded!")
            st.dataframe(st.session_state.data.head())

# Page 2: EDA
elif page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    if st.session_state.data is None:
        st.warning("Please upload or generate a dataset in the 'Dataset Upload' page.")
    else:
        data = st.session_state.data
        st.subheader("Data Overview")
        st.dataframe(data.head())
        st.write("Shape:", data.shape)
        st.subheader("Summary Statistics")
        st.dataframe(data.describe())
        st.subheader("Missing Values")
        st.dataframe(data.isnull().sum())
        st.subheader("Duplicates")
        st.write("Number of duplicates:", data.duplicated().sum())
        st.subheader("Correlation Matrix")
        numeric_data = data.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.subheader("Average Revenue by Category")
        avg_rev = data.groupby('category')['ad_revenue_usd'].mean()
        fig2, ax2 = plt.subplots()
        avg_rev.plot(kind='bar', ax=ax2)
        st.pyplot(fig2)

# Page 3: Preprocessing
elif page == "Preprocessing":
    st.header("Data Preprocessing")
    if st.session_state.data is None:
        st.warning("Please upload or generate a dataset in the 'Dataset Upload' page.")
    else:
        data = st.session_state.data.copy()
        st.subheader("Original Data")
        st.dataframe(data.head())

        # Remove duplicates
        data = data.drop_duplicates()
        st.write("Duplicates removed. New shape:", data.shape)

        # Handle missing values
        for col in ['views', 'likes', 'comments', 'watch_time_minutes']:
            data[col] = data[col].fillna(data[col].median())

        # Feature Engineering
        data['engagement_rate'] = (data['likes'] + data['comments']) / data['views']
        data['year'] = pd.to_datetime(data['date']).dt.year
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['day'] = pd.to_datetime(data['date']).dt.day
        data = data.drop(['video_id', 'date'], axis=1)

        # Encode categoricals (done in pipeline later, but show processed data)
        st.subheader("Processed Data")
        st.dataframe(data.head())
        st.session_state.processed_data = data
        st.success("Preprocessing complete! Proceed to Data Visualization or Model Training.")

# Page 4: Data Visualization
elif page == "Data Visualization":
    st.header("Data Visualization")
    if st.session_state.processed_data is None:
        st.warning("Please complete Preprocessing first.")
    else:
        data = st.session_state.processed_data
        st.subheader("Ad Revenue Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['ad_revenue_usd'], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Boxplot of Views")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=data['views'], ax=ax2)
        st.pyplot(fig2)

        st.subheader("Revenue by Category")
        fig3, ax3 = plt.subplots()
        sns.barplot(x='category', y='ad_revenue_usd', data=data, ax=ax3)
        st.pyplot(fig3)

        st.subheader("Correlation Heatmap")
        numeric_data = data.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        fig4, ax4 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)

# Page 5: Model Training
elif page == "Model Training":
    st.header("Model Training")
    if st.session_state.processed_data is None:
        st.warning("Please complete Preprocessing first.")
    else:
        data = st.session_state.processed_data
        X = data.drop('ad_revenue_usd', axis=1)
        y = data['ad_revenue_usd']
        cat_cols = ['category', 'device', 'country']
        num_cols = [col for col in X.columns if col not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ])

        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }

        if st.button("Train Models"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            results = {}
            for name, model in models.items():
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
            st.session_state.model_results = results
            best_name = max(results, key=lambda x: results[x]['R2'])
            best_model = models[best_name]
            best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
            best_pipeline.fit(X, y)
            joblib.dump(best_pipeline, 'best_model.pkl')
            st.session_state.model = best_pipeline
            st.success(f"Models trained! Best model: {best_name}")

            # Display results in table
            st.subheader("Model Evaluation Metrics")
            st.table(pd.DataFrame(results).T)

# Page 6: Prediction
elif page == "Prediction":
    st.header("Prediction")
    if st.session_state.model is None:
        st.warning("Please train the model in the 'Model Training' page.")
    else:
        model = st.session_state.model
        # User inputs (same as before)
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

        engagement_rate = (likes + comments) / views if views > 0 else 0

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

        if st.button('Predict Revenue'):
            prediction = model.predict(input_data)[0]
            st.success(f'Predicted Ad Revenue: ${prediction:.2f}')