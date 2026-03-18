# 🏦 Universal Bank — Personal Loan Campaign Intelligence Dashboard

A comprehensive Streamlit dashboard for predicting personal loan acceptance using classification algorithms, with descriptive-to-prescriptive analytics for hyper-personalised marketing campaigns.

## 📁 Project Structure

```
universal_bank_dashboard/
├── app.py                  # Main Streamlit application
├── UniversalBank.csv       # Training dataset (5,000 records)
├── sample_test_data.csv    # Sample test file for predictions (200 records)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
└── README.md               # This file
```

## 🚀 Getting Started

### Local Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd universal_bank_dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### Deploy on Streamlit Cloud

1. Push this entire folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select the repo.
4. Set the main file path to `app.py`.
5. Click **Deploy**.

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Executive Summary** | KPI cards, target distribution, income/education/family analysis, banking relationships |
| **Customer Deep-Dive** | Correlation heatmap, income segmentation, education × income interaction, CD account analysis |
| **Model Performance** | Decision Tree, Random Forest & Gradient Boosted Tree metrics table, single ROC curve, confusion matrices, feature importance |
| **Campaign Strategy** | Ideal customer profile, segment priority matrix, tiered budget allocation, actionable recommendations |
| **Predict New Customers** | Upload test CSV → get predictions with acceptance probability & priority tags → download results |

## 🤖 Models Used

- **Decision Tree** — Interpretable baseline model
- **Random Forest** — Ensemble of decision trees for improved accuracy
- **Gradient Boosted Tree** — Sequential boosting for best overall performance

## 📥 Using the Prediction Feature

1. Navigate to **Predict New Customers** page.
2. Upload a CSV with these columns: `ID, Age, Experience, Income, ZIP Code, Family, CCAvg, Education, Mortgage, Securities Account, CD Account, Online, CreditCard`.
3. The model predicts `Personal Loan` and assigns a priority label (HIGH / MEDIUM / LOW).
4. Download the result CSV with predictions.

A `sample_test_data.csv` file is included for testing.

## 📋 Column Descriptions

| Column | Description |
|--------|-------------|
| ID | Customer ID |
| Age | Customer age in years |
| Experience | Years of professional experience |
| Income | Annual income ($000) |
| ZIP Code | Home address ZIP code |
| Family | Family size |
| CCAvg | Average monthly credit card spending ($000) |
| Education | 1: Undergrad, 2: Graduate, 3: Advanced/Professional |
| Mortgage | House mortgage value ($000) |
| Personal Loan | Target — 1: Accepted, 0: Not accepted |
| Securities Account | 1: Has securities account |
| CD Account | 1: Has certificate of deposit account |
| Online | 1: Uses internet banking |
| CreditCard | 1: Uses bank credit card |

## 🛠 Tech Stack

- Python 3.9+
- Streamlit
- Pandas & NumPy
- Scikit-learn
- Plotly
