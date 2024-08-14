from error_code import error
import pandas as pd
df = pd.read_excel("C:/Users/pavan/OneDrive/Desktop/data_for_test.xlsx")
# Convert these columns to numpy arrays
P = df['T'].to_numpy()
T = df['T+1'].to_numpy()
# Calculate error metrics
results = error(P, T)
for key, value in results.items():
    print(f"{key}: {value}")
