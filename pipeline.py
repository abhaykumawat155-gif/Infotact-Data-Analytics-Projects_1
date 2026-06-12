import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("🚀 Starting Consumer360 Data Pipeline...")

# ==========================================
# 1. DATA GENERATION (Simulating SQL/Raw Data)
# ==========================================
np.random.seed(42)
num_rows = 1500

customer_ids = [f"CUST-{np.random.randint(1000, 1080)}" for _ in range(num_rows)]
start_date = datetime(2025, 1, 1)
dates = [start_date + timedelta(days=int(np.random.randint(0, 360))) for _ in range(num_rows)]
amounts = np.random.uniform(10, 600, num_rows).round(2)

# Faulty rows integration to simulate real cleansing task (NULLs and negative values)
amounts[np.random.choice(num_rows, 30, replace=False)] = -100.0 
customer_ids_with_nan = [np.nan if i % 50 == 0 else customer_ids[i] for i in range(num_rows)]

raw_df = pd.DataFrame({
    'CustomerID': customer_ids_with_nan,
    'OrderDate': dates,
    'TotalAmount': amounts
})

# ==========================================
# 2. DATA CLEANING & TRANSFORMATION (ETL)
# ==========================================
# Handle missing attributes and remove negative metrics (returns)
cleaned_df = raw_df.dropna(subset=['CustomerID']).copy()
cleaned_df = cleaned_df[cleaned_df['TotalAmount'] > 0]
cleaned_df['OrderDate'] = pd.to_datetime(cleaned_df['OrderDate'])

print("✅ Data Ingestion & ETL Completed.")

# ==========================================
# 3. RFM SEGMENTATION ENGINE
# ==========================================
snapshot_date = cleaned_df['OrderDate'].max() + pd.Timedelta(days=1)

rfm = cleaned_df.groupby('CustomerID').agg({
    'OrderDate': lambda x: (snapshot_date - x.max()).days,
    'CustomerID': 'count',
    'TotalAmount': 'sum'
}).rename(columns={'OrderDate': 'Recency', 'CustomerID': 'Frequency', 'TotalAmount': 'Monetary'})

# Scoring parameters allocation (1-5 scale)
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

# Structural customer segmentation tiers mapping
def assign_segment(row):
    r, f = row['R_Score'], row['F_Score']
    if r >= 4 and f >= 4: return 'Champions'
    elif r >= 3 and f >= 3: return 'Loyalists'
    elif r <= 2 and f >= 3: return 'At Risk / Churn Risk'
    else: return 'Hibernating'

rfm['Segment'] = rfm.apply(assign_segment, axis=1)

# Exporting directly inside your current folder
rfm.to_csv('rfm_bi_export.csv')
print("✅ Project Data successfully saved as 'rfm_bi_export.csv'!")