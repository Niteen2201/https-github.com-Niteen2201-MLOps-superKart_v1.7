import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset from the correct path inside the repo
df = pd.read_csv("data/SuperKart.csv")

# Drop unique identifier column that won't help in prediction
if "Product_Id" in df.columns:
    df = df.drop(columns=["Product_Id"])

# Target is the numerical column Product_Store_Sales_Total for regression
X = df.drop(columns=["Product_Store_Sales_Total"])
y = df["Product_Store_Sales_Total"]

# Split without stratify as this is a regression task
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save the local training and testing splits
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
yitrain = pd.DataFrame(ytrain) # convert Series to DataFrame to save
yitrain.to_csv("ytrain.csv", index=False)
yitest = pd.DataFrame(ytest)
yitest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written for regression.")
