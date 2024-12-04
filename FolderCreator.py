import os

# List of companies
companies = ["Apple", "NVIDIA", "Microsoft", "Alphabet Inc. Class A", "Alphabet Inc. Class C", "Meta Platforms", "Adobe", "Oracle", "Intel", "Qualcomm", "Texas Instruments", "Salesforce", "ServiceNow"]  # Add the companies you want

# Years and months
years = range(2015, 2025)  # From 2015 to 2024
months = range(1, 13)      # Months 1 to 12

# Create the directories
for company in companies:
    for year in years:
        for month in months:
            # Construct the path
            path = os.path.join(company, str(year), str(month))
            # Create the directory (including any missing intermediate ones)
            os.makedirs(path, exist_ok=True)

print("Directories created successfully!")