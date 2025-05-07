import pandas as pd
from sklearn.tree import DecisionTreeRegressor as dtr
data=pd.read_csv("Bangalore.csv")
data['Whitefield'] = data['Location'].apply(lambda loc: 1 if loc.strip().lower() == 'whitefield' else 0)
#one hot encoding for location data considering most expensive as 1 and rest as 0
data['JPNagar'] = data['Location'].apply(lambda loc: 1 if loc.strip().lower() == 'jp nagar' else 0)

# Define target and features
y = data['Price']
features = ['Area', 'No. of Bedrooms', 'Resale', 'Whitefield','JPNagar']
x = data[features]

# Train the model
model = dtr(random_state=1)
model.fit(x, y)

# Collect user input
a = int(input("Enter area in sq ft: "))
n = int(input("Enter number of bedrooms/BHK: "))
r = input("Has it been resold? (Yes/No): ")
if r.lower() == "yes":
    r = 1
else:
    r = 0
l = input("Enter the location of the house: ")
if l == 'whitefield':
    whitefield = 1
    jp_nagar = 0
elif l == 'jp nagar':
    whitefield = 0
    jp_nagar = 1
else:
    whitefield = 0
    jp_nagar = 0

# Create a DataFrame with the input data
inp = pd.DataFrame([{
    'Area': a,
    'No. of Bedrooms': n,
    'Resale': r,
    'Whitefield': whitefield,
    'JPNagar': jp_nagar
}])

# Make prediction
prediction = model.predict(inp)
print(f"The cost will be around Rs {prediction[0]:,.2f}")
