
from sklearn.datasets import make_regression
import plotly.express as px
import pandas as pd

# Generate a synthetic dataset with 100 samples, 2 features, and some noise
X, y = make_regression(n_samples=100, n_features=10, noise=0.1)

# Create a DataFrame from the data
#df = pd.DataFrame(data=X, columns=['Feature 1', 'Feature 2'])
#df['Target'] = y

print(X)
print(y)
    
# Create a 3D scatter plot of the data
#fig = px.scatter_3d(df, x='Feature 1', y='Feature 2', z='Target')
#fig.show()
