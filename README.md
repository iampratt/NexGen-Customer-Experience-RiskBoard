# NexGen Logistics — Customer Experience Risk Dashboard

Predict at‑risk orders and surface targeted interventions to improve customer experience, reduce churn, and control costs.

## Problem & Business Justification

NexGen faces delivery delays, quality issues, and cost pressures. These drive poor CX (low ratings, complaints) and churn. The dashboard transforms operations from reactive to proactive by scoring risk early and recommending actions. This supports 15–20% cost reduction via fewer re‑deliveries/contacts and better carrier/route choices.

## Datasets (CSV, project root)

- `orders.csv`: Order_ID, Order_Date, Customer_Segment, Priority, Product_Category, Order_Value_INR, Origin, Destination, Special_Handling
- `delivery_performance.csv`: Carrier, Promised/Actual_Delivery_Days, Delivery_Status, Quality_Issue, Customer_Rating, Delivery_Cost_INR
- `routes_distance.csv`: Route, Distance_KM, Fuel_Consumption_L, Toll_Charges_INR, Traffic_Delay_Minutes, Weather_Impact
- `customer_feedback.csv`: Feedback_Date, Rating, Feedback_Text, Would_Recommend, Issue_Category
- `cost_breakdown.csv`: Fuel, Labor, Vehicle_Maintenance, Insurance, Packaging_Cost, Technology_Platform_Fee, Other_Overhead
- `vehicle_fleet.csv`, `warehouse_inventory.csv`: contextual (not required for v1 risk model)

Place all CSVs in the project root. The app automatically loads from the working directory.

## Features & Engineering

Derived features include:

- Timeliness: `delay_days = max(Actual−Promised, 0)`, severe/slight delay flags
- Quality: one‑hot `Quality_Issue` and `Delivery_Status`
- Cost: `cost_ratio = Delivery_Cost_INR / Order_Value_INR`
- Route stress: `Traffic_Delay_Minutes`, weather impact flags (Fog/Heavy_Rain)
- Voice of customer: `Customer_Rating`, feedback rating, simple negative sentiment from text (rule‑based lexicon)
- Context: Segment, Priority, Product Category, Carrier, Corridor
- Missing handling: numeric median impute + missing indicators; categorical mode/“Unknown”

## Risk Scoring

Hybrid risk score blends a heuristic (rules) and an ML probability.

### Heuristic (0–100)

Weights sum to 1; each subscore normalized to 0..1:

- 0.40 × delivery delay score (greater delay → higher risk)
- 0.30 × average of low rating score and negative text sentiment
- 0.15 × quality issues (Wrong_Item / Damage)
- 0.10 × cost ratio (high cost vs value)
- 0.05 × route stress (traffic + adverse weather)

`risk_heuristic = 100 × Σ(w_i × s_i)`

### ML Model (probability → 0..100)

- Model: Logistic Regression (balanced class weights)
- Target (binary at‑risk): `rating ≤ 2` OR `Delivery_Status == Severely-Delayed` OR `Quality_Issue in {Wrong_Item, Minor_Damage, Major_Damage}`
- Inputs: numeric (delays, costs, traffic, sentiment) + one‑hot categorical (status, issue, weather, segment, priority, product, carrier)
- Time‑aware split: last 20% for proxy test
- Metrics (Overview → Model metrics): ROC‑AUC, precision/recall/F1 @ threshold 0.65

### Blending and Bands

`risk_final = 0.5 × risk_heuristic + 0.5 × (100 × p_ml)`

Risk bands:

- Low < 35
- Medium 35–65
- High > 65

## Suggested Interventions

Context‑aware actions in the Orders view detail drawer:

- Any quality issue → compensation credit + priority re‑ship
- Severely delayed or delay ≥2 days → proactive outreach + ops escalation
- Slight delay or delay ≥1 → ETA update + monitoring
- Express with any delay → carrier/route upgrade
- High risk (>65) → add credit + post‑delivery CSAT follow‑up; Medium (35–65) → enable proactive notifications

## Visualizations

- Avg risk by segment (bar)
- Weekly delay and CSAT trend (line)
- Cost vs delay colored by risk (scatter)
- Quality issues distribution (bar)

## Model Explainability

- Coefficient importances (top weighted features) for the LR model
- Metrics on recent 20% test split: ROC‑AUC, precision/recall/F1 at high‑risk threshold

## Business Impact Calculator

In Overview, estimate retained revenue from proactive contacts: inputs = weekly contacts, save rate %, avg order value, credit/compensation, horizon (weeks). Shows gross retained revenue, credits cost, and net benefit.

## Usage

1. Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure the 7 CSVs are in the project root.
3. Run

```
streamlit run app/main.py
```

## Deployment

- Local/VM: run the command above. The app reads from the working directory.
- Docker (example):

```
docker run -p 8501:8501 \
  -v /absolute/host/path/to/csvs:/app \
  your-image:tag
```

Open `http://localhost:8501`.

## Roadmap (v2 Enhancements)

- Additional models: Gradient Boosting + calibrated probabilities; simple LR+GB ensemble
- Richer features: carrier/route history, VADER sentiment fallback
- Explainability: permutation importances; per‑order contribution summaries
- UX: PNG chart export; What‑If intervention simulator; threshold tuner and PR‑AUC/confusion matrix

## Code Structure

- `app/main.py`: Streamlit entry, filters, routing, downloads
- `app/data_loader.py`: CSV loading and schema normalization
- `app/features.py`: feature engineering and missing‑data handling
- `app/model.py`: LR pipeline, risk blending, metrics, importances
- `app/views/overview.py`: KPIs, charts, metrics, impact calculator
- `app/views/segment.py`: segment drilldown, cohorts
- `app/views/orders.py`: orders table, detail actions, exports
- `app/components/charts.py`: Plotly chart helpers

## Notes

- Partial data across files is expected; loaders impute and add missing flags.
- If metrics show low AUC, review label definition and thresholds or consider adding more historical features (see Roadmap).
