![screenshot](https://github.com/Mainabryan/TUNUSIAN-BILLING-ANALYSIS/blob/d9a48566aa7dfedbc2acc1d9a4a6fe7075929196/Screenshot%202025-07-20%20075905.png)
# Tunisian Billing Analysis Web App

This Streamlit application provides an interactive interface for exploring and analyzing electricity billing data from Tunisia. It includes data visualization and predictive modeling features to help identify consumption patterns and forecast future usage.

---

## 📊 Dataset Overview

The dataset contains information about electricity bills for clients in Tunisia. Key columns include:

- `client_id`: Unique identifier for each customer
- `invoice_date`: Date the bill was issued
- `tarif_type`: Type of tariff plan
- `counter_number`: Meter ID
- `counter_code`: Code related to the meter
- `counter_coefficient`: Coefficient for scaling consumption
- `consommation_level_1` to `consommation_level_4`: Electricity consumption in different levels
- `old_index`, `new_index`: Meter readings before and after
- `months_number`: Billing duration in months
- `counter_type`: Type of counter
- `counter_type_encoded`: Numeric encoding of counter type

---

## 🚀 Features

- Interactive data exploration using Streamlit
- Visualizations: line charts, bar graphs, histograms
- Predictive model for estimating total consumption
- Downloadable results and input-friendly interface

---

## 🔧 Installation

1. Clone the repository or download the files
2. Navigate to the project folder
3. Install the requirements:

```bash
pip install -r requirements.txt
