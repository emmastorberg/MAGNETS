import pandas as pd

# Read the CSV file
df = pd.read_csv('/home/emmas/MAGNETS/veas_extended_pilot_data copy.csv')

# Convert date column to datetime (adjust 'date_column' to your actual date column name)
df['Time'] = pd.to_datetime(df['Time'])

# Filter data before January 1st, 2025
df_shortened = df[df['Time'] < '2025-01-01']

# Save the shortened data
df_shortened.to_csv('/home/emmas/MAGNETS/veas_extended_pilot_data copy.csv', index=False)

print(f"Data shortened. Rows before 2025-01-01: {len(df_shortened)}")