import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Calculate correlation matrix and select top features
corr_matrix = data.corr()
top_features = corr_matrix.nlargest(10, 'Label')['Label'].index
selected_data = data[top_features]

# Save selected features
selected_data.to_csv('selected_features.csv', index=False)

# Plot correlation heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.show()
