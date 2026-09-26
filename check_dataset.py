import pandas as pd

# Load dataset
df = pd.read_csv("data/startup data.csv")

print("=" * 50)
print("STARTUP DATASET INFORMATION")
print("=" * 50)

# Number of rows and columns
print("\n1. Dataset Shape:")
print(df.shape)

# Column names
print("\n2. Column Names:")
for i, column in enumerate(df.columns, 1):
    print(i, ":", column)

# First 5 rows
print("\n3. First 5 Rows:")
print(df.head().to_string())

# Data types
print("\n4. Data Types:")
print(df.dtypes)

# Missing values
print("\n5. Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0])

# Check is_top500 values
print("\n6. is_top500 Values:")
print(df["is_top500"].value_counts(dropna=False))
print("\n6. is_top500 Values:")
print(df["is_top500"].value_counts(dropna=False))

# Check status values
print("\n7. Status Values:")
print(df["status"].value_counts(dropna=False))

print("\n" + "=" * 50)