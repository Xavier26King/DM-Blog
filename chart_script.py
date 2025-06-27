import pandas as pd
import plotly.graph_objects as go

# Load the data
df = pd.read_csv('pagerank_iterations.csv')

# Create the line chart
fig = go.Figure()

# Brand colors for the 4 pages
colors = ['#1FB8CD', '#FFC185', '#ECEBD5', '#5D878F']
page_columns = ['page_A', 'page_B', 'page_C', 'page_D']
page_labels = ['A', 'B', 'C', 'D']

# Add a line for each page
for i, (col, label) in enumerate(zip(page_columns, page_labels)):
    fig.add_trace(go.Scatter(
        x=df['iteration'],
        y=df[col],
        mode='lines+markers',
        name=label,
        line=dict(color=colors[i], width=3),
        marker=dict(size=6),
        cliponaxis=False
    ))

# Update layout
fig.update_layout(
    title="PageRank Convergence Over Iterations",
    xaxis_title="Iteration",
    yaxis_title="PageRank Val",
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Set axis ranges
fig.update_xaxes(range=[-0.5, 14.5])
fig.update_yaxes(range=[0, 0.4])

# Save the chart
fig.write_image("pagerank_convergence.png")
print("Chart saved successfully as pagerank_convergence.png")