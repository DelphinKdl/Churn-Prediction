# Customer Churn Prediction System

## A Machine Learning Application that predicts customer churn for telecom companies, enabling proactive retention strategies and reducing revenue loss.

![ML System Design](images/Churn-prediction%20architecture.png)

## Executive Summary

This **Customer Churn Prediction System** solves a critical business problem for telecom providers managing millions of subscribers. By implementing a CatBoost Classifier optimized with Optuna, I achieved an F1-score of 0.72, providing a 0.64 precision on the churn class. This allows the business to transition from reactive support to proactive retention, targeting the specific segments (Month-to-Month and Fiber Optic users) that drive the majority of revenue leakage.

## Business Problem Solved

**Challenge**: Telecom providers face high attrition rates that stabilize or decrease total ARR (Annual Recurring Revenue). The business needs to:
- **Improving customer retention**: Identifying high-risk users early before they cancel our services.
- **Reducing churn-driven revenue loss**: Minimize the high cost of acquiring new customers vs. retaining existing ones 
- **Supporting targeted interventions** Apply discounts and offers only to those predicted to leave.

**Key Findings & Data Observations**:
- **The Contract Trap**: Customers on **Month-to-Month** contracts dominate churn. 1-year and 2-year contracts are the most powerful retention factors.
- **Service Stickiness**: Lack of **OnlineSecurity and TechSupport** are the strongest technical indicators of churn.
- **High-Value Risk**: **Fiber Optic** users and those with **high monthly charges** ($80–$90) churn more than DSL users, suggesting a gap in price-to-value perception.
- **Payment Friction**: **Electronic check** users churn significantly more than those using automated bank transfers or credit cards
- **The Tenure Effect**: Churn is heavily concentrated in the first **0-12 months**. If a customer stays past the first year, their churn probability drops drastically.


**Solution**: A batch ML system that scores every customer's churn probability, enabling:
- **Proactive Retention Campaigns**: Personalized outreach to high-risk customers
- **Contract & Pricing Optimization**: Data-driven incentives (e.g., long-term contract offers, bundled services)
- **Payment Method Incentives**: Offer a small one-time credit for users who switch from **Electronic Check to Auto-pay** (Bank Transfer/Credit Card) to reduce involuntary churn.

## Model Performance
I utilized **Minority Class Upsampling** to address the 73/27 class imbalance in the dataset, ensuring the model could accurately learn churn patterns without losing majority-class data.

| Model | Accuracy | F1 (Macro Avg) | F1 (Churn Class) | Precision (Churn) | Recall (Churn) |
|-------|----------|----------------|-------------------|--------------------|----------------|
| Random Forest (Baseline) | 0.79 | 0.70 | 0.55 | 0.62 | 0.49 |
| Random Forest + Upsampling | 0.78 | 0.72 | 0.58 | 0.59 | 0.57 |
| **CatBoost + Optuna (Final)** | **0.79** | **0.72** | **0.57** | **0.64** | **0.52** |

**Engineering Decision**: I selected the **CatBoost + Optuna** model for production because it offers the highest Precision (0.64). In a business context, this ensures retention marketing budgets are spent on the customers most likely to actually leave, minimizing "waste" on customers who would have stayed regardless.

---
## Visual Intelligence (Power BI)
To translate model findings into stakeholder-ready insights, I developed a 4-page interactive dashboard focused on Customer Retention Analytics.

- **Executive Overview**: High-level KPI monitoring for management 
- **Customer Segmentation & Risk Profile**: Identified that customers in their first 6 months represent 40% of all churn events, marking the "Onboarding Phase" as the highest risk period.
- **Service & Product Analysis**: Evaluation of product "stickiness" and service gaps.


![Executive Overview](images/Churn-Overview.png)
![Customer Segmentation & Risk Profile](images/Churn-Customer-Segmentation.png)
![Service & Product Analysis](images/Churn-Service.png)

**Strategic Recommendations**: 
Based on the combined ML findings and Power BI analysis, I recommend the following:
- **Contract Migration**: Launch automated campaigns to transition **Month-to-Month** users into 1-year contracts, as this is the highest lever for reducing attrition
- **Service Bundling**: Proactively offer OnlineSecurity or TechSupport trials to high-risk Fiber Optic users to increase service engagement.
- **Operational Insights**: Identification of the strongest churn drivers - contract type, payment method, and service engagement

### Features in Production Churn Systems

- **Recency**: Days since last login or activity
- **Frequency**: Number of logins, sessions, or purchases in the last X days
- **Engagement duration**: Average session length over the last week or month
- **Feature usage counts**: How often key features were used (e.g., reports generated, messages sent)
- **Plan or tier info**: Subscription type, feature access
- **Support activity**: Number of support tickets filed, time to resolution
- **Billing patterns**: Failed payments, payment method changes, recent upgrades/downgrades
- **Marketing interaction**: Click-through rate on emails, offers redeemed
- **Net promoter score (NPS)** or survey feedback when available
- **Account age**: Days since sign-up or subscription
- **Inactivity streak**: Longest stretch of inactivity in recent time window

### Target

Churn / non-churn label within a defined churn window (e.g., next 14 days)

### Models

- **Random Forest and Gradient Boosting** (XGBoost, LightGBM, CatBoost)
- **Logistic Regression** for interpretable binary classification
- **Neural Networks** for behavioral and event sequence modeling
- **Autoencoders or Isolation Forests** for anomaly-based churn risk


## Application Architecture

The architecture implements a production-grade ML system with two main flows:

![ML System Design](images/Churn-prediction%20architecture.png)

### Data Sources
- **CRM** (PostgreSQL) - customer profiles and account data
- **Usage logs** (Snowflake, ClickHouse) - behavioral and session data
- **Billing Data** (SaaS Platforms) - payment and subscription data
- **Support tickets** (SaaS Platforms) - customer support interactions

### Offline (Batch) Training
Data from all sources flows through an **ETL Pipeline** into modular ML stages:
1. **Preprocessing Pipeline** → 2. **Feature Engineering Pipeline** → 3. **Training Pipeline** → 4. **Postprocessing Pipeline**

Training is triggered by:
1. Weekly / Monthly schedule
2. Model performance degradation
3. Data drift detection

Models, metrics, and metadata are stored in a **Model Storage / Registry** (MLflow / Comet).

### Batch Inference
A **(usually) daily trigger** pulls fresh customer data through the same ETL, Preprocessing, and Feature Engineering pipelines, then runs the **Inference Pipeline**. The **Postprocessing Pipeline** outputs churn/non-churn scores to a PostgreSQL database (`customer_id: churn_risk`). High-risk customers automatically trigger **alerts** and **personalized outreach actions**.

---

## End-to-End ML Pipeline

#### 1. **Preprocessing Pipeline**
- Removal of zero-tenure customers (never used the service)
- Cleaning of `TotalCharges` column (empty strings → NaN → float conversion)
- Null value handling and data type standardization

#### 2. **Feature Engineering Pipeline**
- **Label Encoding** of categorical features (optimized for tree-based models, avoids one-hot sparsity)
- 19 features across 5 categories: demographics, account/billing, service subscriptions, contract details, and payment information

#### 3. **Training Pipeline**
- **Baseline**: Random Forest Classifier (100 estimators)
- **Upsampling**: Minority class resampling to address 73/27 class imbalance
- **CatBoost Classifier**: Gradient boosting with **Optuna hyperparameter tuning** (150 trials)
  - 5-Fold Stratified Cross-Validation
  - Early stopping (100 rounds)
  - Optimized: learning_rate, depth, l2_leaf_reg
- **Best hyperparameters**: `learning_rate=0.111`, `depth=4`, `l2_leaf_reg=1.05`, `iterations=139`

#### 4. **Inference Pipeline**
- Model loading and batch prediction on new customer data
- Outputs churn probability per customer

#### 5. **Postprocessing Pipeline**
- Model persistence and metric logging
- Churn risk scores written to the database for downstream alerting

---


## Quick Start

```bash
# Clone the repository
git clone https://github.com/DelphinKdl/Vodafone-customer-churn-ML-prediction.git
cd Churn-Prediction

# Create and activate virtual environment
python -m venv churn
source churn/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook churn_prediction.ipynb
```

---

## Project Architecture & Data Flow

```
Churn-Prediction/
├── Data/
│   └── Telco-Customer-Churn.csv       # 7,043 customers × 21 features
├── Doc/
│   └── Churn Prediction.pdf           # Project design document
├── images/
│   └── Churn-prediction architecture.png
├── churn_prediction.ipynb             # End-to-end ML notebook (EDA → Modeling → Evaluation)
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Tech Stack

- **ML Frameworks**: CatBoost, Scikit-learn, XGBoost
- **Hyperparameter Tuning**: Optuna (150 trials, TPE sampler)
- **Data Analysis**: Pandas, NumPy, YData Profiling
- **Visualization**: Matplotlib, Seaborn, Plotly, Power BI (coming soon)
- **Experiment Tracking**: MLflow
- **Language**: Python 3.11

---

## Roadmap

- [ ] **Interactive Power BI Dashboard** - Visual analytics layer for churn insights, customer segmentation, and retention KPIs

---

## License

This project is licensed under a custom **Personal Use License**.

You are free to:
- Use the code for personal or educational purposes
- Publish your own fork or modified version on GitHub **with attribution**

You are **not allowed to**:
- Use this code or its derivatives for commercial purposes
- Resell or redistribute the code as your own product
- Remove or change the license or attribution

For any use beyond personal or educational purposes, please contact the author for written permission.
