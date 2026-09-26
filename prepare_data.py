import pandas as pd

# ==========================================
# 1. Load the original dataset
# ==========================================

df = pd.read_csv("data/startup data.csv")

print("Original dataset shape:", df.shape)


# ==========================================
# 2. Create the target variable
# ==========================================

# acquired = 1 (Success)
# closed   = 0 (Failure)

df["success"] = df["status"].map({
    "acquired": 1,
    "closed": 0
})


# ==========================================
# 3. Remove unnecessary and leakage columns
# ==========================================

columns_to_drop = [
    "Unnamed: 0",
    "Unnamed: 6",
    "id",
    "object_id",
    "name",
    "city",
    "zip_code",
    "closed_at",
    "status",
    "is_top500",
    "founded_at",
    "first_funding_at",
    "last_funding_at"
]

df = df.drop(columns=columns_to_drop, errors="ignore")


# ==========================================
# 4. Display remaining columns
# ==========================================

print("\nRemaining columns:")
print(df.columns.tolist())


# ==========================================
# 5. Check target values
# ==========================================

print("\nTarget distribution:")
print(df["success"].value_counts())


# ==========================================
# 6. Save the prepared dataset
# ==========================================

df.to_csv("data/prepared_startup_data.csv", index=False)

print("\nPrepared dataset saved successfully!")
print("New dataset shape:", df.shape)