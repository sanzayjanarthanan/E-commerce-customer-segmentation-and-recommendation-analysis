# E-Commerce Customer Segmentation & Recommendation Analysis

**Veda Technology · AI & ML Track · Final Project**

## Overview
This project analyses customer purchasing behaviour from an online retail dataset and uses unsupervised machine learning (K-Means) to group customers into meaningful business segments. A basic recommendation component then suggests products using cosine similarity. The goal is to show how raw customer behaviour can be converted into actionable business intelligence.

## Tools Used
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · K-Means · Cosine Similarity · Jupyter Notebook / Google Colab

## Dataset
**UCI Online Retail dataset**: transactions of a UK-based online retailer between December 2010 and December 2011.

| Column | Description |
|--------|-------------|
| InvoiceNo | Invoice number (starts with "C" for cancellations) |
| StockCode | Product code |
| Description | Product name |
| Quantity | Units per transaction |
| InvoiceDate | Date and time of the invoice |
| UnitPrice | Price per unit |
| CustomerID | Customer identifier |
| Country | Customer country |

Raw size: **541,909 rows × 8 columns** *(confirm with the `Raw shape` line in your output)*.

## Files in this Repository
| File | Description |
|------|-------------|
| `customer_segmentation.ipynb` | Complete notebook / code |
| `cleaned_transactions.csv` | Cleaned transaction dataset |
| `customer_features.csv` | Customer-level feature table (RFM and extra features) |
| `customer_segments.csv` | Each customer with cluster and segment name |
| `segment_profiles.csv` | Business profile of each segment |
| `rfm_distributions.png` | RFM distributions before and after log transform |
| `elbow_silhouette.png` | Elbow method and silhouette analysis |
| `customer_segments_2d.png` | 2D cluster visualization (PCA) |
| `recency_vs_monetary.png` | Recency vs Monetary by segment |
| `README.md` | Approach and results |

## Approach

### 1. Data Cleaning
| Step | Reason |
|------|--------|
| Dropped rows with missing `CustomerID` | Customer-level analysis needs a customer to attach the purchase to |
| Removed cancelled invoices (`InvoiceNo` starting with "C") | They are returns, not purchases |
| Removed rows with quantity ≤ 0 or unit price ≤ 0 | Invalid or adjustment records |
| Removed non-product codes (postage, fees, manual entries) | They are not products customers chose |
| Removed exact duplicate rows | Repeated records would inflate frequency and spend |
| Converted types (`InvoiceDate` to datetime, `CustomerID` to int) and created `TotalPrice = Quantity × UnitPrice` | Needed for the features |

Result: **<rows before> → <rows after> rows**, **<N> customers**, **<N> products**. *(copy from the `Cleaned rows` output line)*

### 2. Customer-Level Features (RFM)
One row per customer, using a snapshot date of one day after the last invoice in the data:

- **Recency:** days since the customer's last purchase (lower is better)
- **Frequency:** number of distinct invoices
- **Monetary:** total money spent

Extra behavioural features were also stored in the feature table: unique products bought, total items and average order value.

### 3. Feature Scaling and Preprocessing
Frequency and Monetary are heavily right-skewed (a few customers buy far more than the rest). K-Means is distance-based, so these values would dominate the clustering. Therefore:

1. A **log transform** (`log1p`) was applied to the RFM features.
2. The result was **standardised** with `StandardScaler` (mean 0, standard deviation 1).

![RFM distributions](rfm_distributions.png)

### 4. Choosing the Number of Clusters
The elbow method and silhouette score were calculated for k = 2 to 8.

![Elbow and silhouette](elbow_silhouette.png)

| k | Silhouette score |
|---|------------------|
| 2 | <value> |
| 3 | <value> |
| 4 | <value> |
| 5 | <value> |
| 6 | <value> |
| 7 | <value> |
| 8 | <value> |

**Chosen k = <K>.** *(Explain in 1-2 lines. If k = 4 was kept for business-friendly segments even though another k had a higher silhouette score, say so honestly, for example: "k = 4 was chosen because it gives four actionable business segments, while k = X had the highest silhouette score of Y.")*

### 5. K-Means Segmentation
K-Means (`n_init=10`, `random_state=42`) was fitted on the scaled RFM features. Final silhouette score for k = <K>: **<value>**.

### 6. Segment Profiles
Clusters were ranked by an RFM score (low recency, high frequency, high monetary is best) and given business names. The names were checked against the profile table below.

| Segment | Customers | Customer share | Avg recency (days) | Avg frequency (orders) | Avg monetary | Revenue share |
|---------|-----------|----------------|--------------------|------------------------|--------------|---------------|
| Champions | <N> | <%> | <> | <> | <> | <%> |
| Regular Customers | <N> | <%> | <> | <> | <> | <%> |
| Occasional Buyers | <N> | <%> | <> | <> | <> | <%> |
| At-Risk Customers | <N> | <%> | <> | <> | <> | <%> |

*(Copy from the `segment_profiles.csv` or the printed profile table.)*

### 7. 2D Customer Cluster Visualization
PCA reduced the three scaled RFM features to two dimensions so the segments could be plotted. The two components explain **<%>** of the variance.

![Customer segments](customer_segments_2d.png)

![Recency vs Monetary](recency_vs_monetary.png)

### 8. Product Recommendation Component
A customer × product matrix was built (1 if the customer bought the product, 0 otherwise). Two simple functions use **cosine similarity**:

- `recommend_similar_products(stock_code, n)`: **item-based**. Products bought by similar groups of customers are considered similar.
- `recommend_for_customer(customer_id, n)`: **customer-based**. Finds the most similar customers and recommends products they bought that the target customer has not bought yet.

**Example 1: products similar to `<StockCode>` (<Description>)**

| StockCode | Description | Similarity |
|-----------|-------------|------------|
| <> | <> | <> |
| <> | <> | <> |
| <> | <> | <> |

**Example 2: recommendations for customer `<CustomerID>`**

| StockCode | Description | Score |
|-----------|-------------|-------|
| <> | <> | <> |
| <> | <> | <> |
| <> | <> | <> |

*(Copy from the demo output of the notebook.)*

## Business Recommendations for Each Segment

| Segment | Recommended action |
|---------|--------------------|
| **Champions** | Reward them with loyalty perks, early access to new products and referral offers. Avoid heavy discounts, because they already buy often. |
| **Regular Customers** | Grow them with bundle offers and cross-sell suggestions from the recommendation function, to move them towards Champions. |
| **Occasional Buyers** | Activate them with seasonal campaigns, free-shipping thresholds and purchase reminders to raise purchase frequency. |
| **At-Risk Customers** | Win them back with personalised re-engagement emails and limited-time comeback offers before they are lost completely. |

## Key Findings
*(Write 3 findings from your own output, each with a number. Examples of the kind of finding to look for:)*
1. <Which segment is the smallest by customers but largest by revenue share, with the numbers>
2. <How many customers are in the At-Risk segment and what share of customers they are>
3. <One observation from the recommendation examples>

## Limitations
- Segments are based only on RFM behaviour. Product category, country and returns were not used.
- The recommendation component is intentionally basic (binary purchase history, no quantity or rating weighting) and was not evaluated against held-out purchases.
- It cannot recommend for brand-new customers or products with no purchase history (cold start).
- Cancelled invoices were removed, so return behaviour is not reflected in the segments.
- The dataset covers one retailer and one year, so the segments may not generalise to other businesses.

## How to Run
**Google Colab:** upload the notebook (or paste the code into one cell) and run it. The dataset is downloaded automatically from UCI; if that fails, upload `Online Retail.xlsx` when prompted. The last step creates `project_outputs.zip` with all CSV and PNG outputs.

**Local Jupyter:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy openpyxl jupyter
jupyter notebook customer_segmentation.ipynb
```
