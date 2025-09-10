import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib

# Generate synthetic data
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

# Preprocessing
data = data.drop_duplicates()
for col in ['views', 'likes', 'comments', 'watch_time_minutes']:
    data[col] = data[col].fillna(data[col].median())
data['engagement_rate'] = (data['likes'] + data['comments']) / data['views']
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data = data.drop(['video_id', 'date'], axis=1)

X = data.drop('ad_revenue_usd', axis=1)
y = data['ad_revenue_usd']
cat_cols = ['category', 'device', 'country']
num_cols = [col for col in X.columns if col not in cat_cols]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Evaluate models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    results[name] = {'R2': r2_score(y_test, y_pred)}

# Select best model
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = models[best_model_name]
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
best_pipeline.fit(X, y)  # Fit on full data
joblib.dump(best_pipeline, 'best_model.pkl')
print(f"Best model saved: {best_model_name}")