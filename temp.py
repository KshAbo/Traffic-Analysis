import pandas as pd

train = pd.read_csv("data/Train.csv")
test  = pd.read_csv("data/TestInputSegments.csv")

print("TRAIN COLUMNS:")
print(train.columns.tolist())

print("\nTEST COLUMNS:")
print(test.columns.tolist())

print("\nTRAIN SAMPLE:")
print(train.head())
