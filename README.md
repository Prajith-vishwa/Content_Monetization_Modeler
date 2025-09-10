# Content Monetization Modeler

## Overview
The Content Monetization Modeler is a project designed to predict YouTube ad revenue based on video metrics using machine learning. It includes data preprocessing, exploratory data analysis (EDA), visualization, model training, and an interactive Streamlit application with a YouTube-themed interface. The project is developed using Python and can be extended with real-world data for enhanced accuracy.

## Features
- Generate or upload a synthetic dataset with 122,000 rows.
- Preprocess data by handling missing values, duplicates, and feature engineering.
- Perform EDA and generate visualizations (e.g., histograms, boxplots, heatmaps).
- Train multiple machine learning models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting).
- Deploy a Streamlit app with a YouTube theme (red sidebar, white main area) for prediction.
- Save the best model for future use.
- Accompanying PPT presentation for project overview.

## Requirements
- Python 3.11 or higher
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `streamlit`
  - `joblib`

## Installation
1. **Clone the Repository:**
   - If hosted on GitHub, clone it: `git clone <repository-url>`.

2. **Set Up a Virtual Environment:**
   - Open a terminal in the project directory.
   - Create a virtual environment: `python -m venv venv`.
   - Activate it:
     - Windows: `venv\Scripts\activate`.

3. **Install Dependencies:**
   - Run: `pip install -r requirements.txt`.
   - If `requirements.txt` is not present, install manually:
  
   - 
4. **Prepare Data:**
- Ensure a preprocessed dataset (`processed_data.csv`) or synthetic data (`youtube_data.csv`) is available, or generate it via the Streamlit app.

## Usage
### Jupyter Notebook
- Open `Simple_Data_Visualization.ipynb` in Jupyter Notebook or JupyterLab.
- Run all cells to load data and generate visualizations.
- Adjust file paths (e.g., `processed_data.csv`) as needed.

### Streamlit App
1. **Run the App:**
- In the terminal (with virtual environment activated), navigate to the project directory.
- Execute: `streamlit run app.py`.
2. **Navigate the App:**
- Use the red sidebar to switch between pages: Dataset Upload, EDA, Preprocessing, Data Visualization, Model Training, Prediction.
- Upload a CSV or generate synthetic data on the "Dataset Upload" page.
- Train models on the "Model Training" page and predict revenue on the "Prediction" page.




## File Structure
Content Monetization/
│
├── app.py                
├── Simple_Data_Visualization.ipynb  
├── Content_Monetization_Modeler.pptx  
├── youtube_data.csv      
├── processed_data.csv    
├── best_model.pkl        
├── README.md             
└── requirements.txt
