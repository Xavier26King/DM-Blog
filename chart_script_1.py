import plotly.graph_objects as go
import json

# Parse the provided data
data_json = [{"damping_factor": 0.5, "page_A": 0.3, "page_B": 0.2, "page_C": 0.3, "page_D": 0.2}, {"damping_factor": 0.7, "page_A": 0.314815, "page_B": 0.185185, "page_C": 0.314815, "page_D": 0.185185}, {"damping_factor": 0.85, "page_A": 0.324561, "page_B": 0.175439, "page_C": 0.324561, "page_D": 0.175439}, {"damping_factor": 0.9, "page_A": 0.327586, "page_B": 0.172414, "page_C": 0.327586, "page_D": 0.172414}, {"damping_factor": 0.95, "page_A": 0.330508, "page_B": 0.169492, "page_C": 0.330508, "page_D": 0.169492}]

# Extract data for plotting
damping_factors = [item['damping_factor'] for item in data_json]
page_A_values = [item['page_A'] for item in data_json]
page_B_values = [item['page_B'] for item in data_json]
page_C_values = [item['page_C'] for item in data_json]
page_D_values = [item['page_D'] for item in data_json]

# Brand colors in order
colors = ['#1FB8CD', '#FFC185', '#ECEBD5', '#5D878F']

# Create the grouped bar chart
fig = go.Figure()

# Add bars for each page with different styling to distinguish overlapping ones
fig.add_trace(go.Bar(
    name='Page A',
    x=damping_factors,
    y=page_A_values,
    marker=dict(color=colors[0], opacity=0.8),
    cliponaxis=False
))

fig.add_trace(go.Bar(
    name='Page B',
    x=damping_factors,
    y=page_B_values,
    marker=dict(color=colors[1], opacity=0.8),
    cliponaxis=False
))

fig.add_trace(go.Bar(
    name='Page C',
    x=damping_factors,
    y=page_C_values,
    marker=dict(color=colors[2], opacity=0.6, line=dict(color=colors[0], width=2)),
    cliponaxis=False
))

fig.add_trace(go.Bar(
    name='Page D',
    x=damping_factors,
    y=page_D_values,
    marker=dict(color=colors[3], opacity=0.6, line=dict(color=colors[1], width=2)),
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='Impact of Damping Factor on PageRank Values',
    xaxis_title='Damping Factor',
    yaxis_title='PageRank Value',
    barmode='group',
    bargap=0.2,
    bargroupgap=0.1,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update axes
fig.update_yaxes(range=[0.15, 0.35])
fig.update_xaxes(tickmode='array', tickvals=damping_factors)

# Save the chart
fig.write_image('damping_pagerank_chart.png')