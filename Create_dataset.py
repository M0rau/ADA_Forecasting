import pandas as pd

# Load the dataset
WTI_data = pd.read_csv('oil_data.csv', delimiter=';')

# French to English month mapping
french_to_english_months = {
    'janv': 'Jan', 'févr': 'Feb', 'mars': 'Mar', 'avr': 'Apr',
    'mai': 'May', 'juin': 'Jun', 'juil': 'Jul', 'août': 'Aug',
    'sept': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'déc': 'Dec'
}

# Replace French month names with English names in the Date column
WTI_data['Date'] = WTI_data['Date'].replace(french_to_english_months, regex=True)

# Convert 'Date' column to datetime
WTI_data['Date'] = pd.to_datetime(WTI_data['Date'], format='%b %d, %Y', errors='coerce')

# Convert 'Price' column to numeric, replacing commas with dots
WTI_data['Price'] = WTI_data['Price'].str.replace(',', '.').astype(float)

# Create shifted price columns
WTI_data['Price_t+1'] = WTI_data['Price'].shift(-1)
WTI_data['Price_t+7'] = WTI_data['Price'].shift(-5)
WTI_data['Price_t+28'] = WTI_data['Price'].shift(-28)
WTI_data['Price_t+91'] = WTI_data['Price'].shift(-91)

# Drop rows with missing values (optional based on your preference for handling missing data)
WTI_data.dropna(inplace=True)

# Set the 'Date' column as the index of the DataFrame (optional if you want Date as a separate column)
WTI_data.set_index('Date', inplace=True)

# Save the dataset
WTI_data.to_csv('combined_data.csv', index=True)  # Set index=False if you don't want Date as index

# Display the first few rows to verify
print(WTI_data.head())