# Let me check the damping factor data and prepare it properly for the chart
import pandas as pd

# Read the damping factor data
df_damping = pd.read_csv('damping_factor_comparison.csv')
print("Damping factor data:")
print(df_damping)

# Create a JSON string for the chart data
import json

chart_data = df_damping.to_dict('records')
print("\nChart data as JSON:")
print(json.dumps(chart_data, indent=2))