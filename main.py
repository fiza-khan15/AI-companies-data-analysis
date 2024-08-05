import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import csv
import warnings
warnings.warn("ignore")

#Creating DataFrame

df = pd.read_csv('AI_Companies.csv')
#print(df)

# Set the display option to show all columns

pd.set_option('display.max_columns' , None)
print(df)

# We replace the inconcistent values with na values

df = pd.read_csv('AI_Companies.csv' , na_values=['Oct-49','02-Sep'])

print(df)

# Check all the columns

print(df.columns)

#We have an extra unessasary column 'Unnamed:8,9,10'.
#We need to delete this column
df = df.drop(columns=['Unnamed: 8']) #, 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11'])
print(df)

#Rename all the columns

df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ','_')

print(df)

# Check the missing values

print(df.isnull().sum())

# Here we have 1 missing value in website and 14 in location.
# Let's handle this.

df.fillna({'location':'Unknown'} , inplace=True)

print(df)

# Check columns name

print(df.columns.tolist())


# Lets create another column 'categorize_location' to analyze data


def categorize_location(row):
    if row['population'] > 500000:  # Adjust threshold as needed
        return 'Metropolitan'
    elif row['population'] > 100000:
        return 'Suburban'
    else:
        return 'Rural'

df['location_category'] = df.apply(categorize_location, axis=1)

print(df)


# Display the unique values in the column to identify non-numeric values
print(df['average_hourly_rate__per_hr_in_usd'].unique())

# Convert the column to numeric, setting errors='coerce' to handle non-numeric values
df['average_hourly_rate__per_hr_in_usd'] = pd.to_numeric(df['average_hourly_rate__per_hr_in_usd'], errors='coerce')

# Group by 'location_category' and calculate the mean

average_hourly_rate_by_location = df.groupby('location_category')['average_hourly_rate__per_hr_in_usd'].mean()

print(average_hourly_rate_by_location)



# Now visualize the analysis

# Create box plot
sns.boxplot(x='location_category', y='average_hourly_rate__per_hr_in_usd', data=df)
plt.title('Hourly Rates by Location Category')
plt.xlabel('Location Category')
plt.ylabel('Average Hourly Rate')
plt.show()


# Histogram
plt.hist(df['average_hourly_rate__per_hr_in_usd'], bins=range(1, 26), edgecolor='black')
plt.title('Distribution of Average Hourly Rate')
plt.xlabel('Hourly Rate ($ per hour)')
plt.ylabel('Frequency')
plt.show()


# Convert 'number_of_employees' into numeric 

df['number_of_employees'] = pd.to_numeric(df['number_of_employees'],errors='coerce')



# Categorize company size (example)
def categorize_company_size(row):
    if row['number_of_employees'] <= 249:
        return 'Small'
    elif row['number_of_employees'] <= 500:
        return 'Medium'
    else:
        return 'Large'

df['company_size'] = df.apply(categorize_company_size, axis=1)

# Calculate average hourly rate by company size
average_hourly_rate_by_size = df.groupby('company_size')['average_hourly_rate__per_hr_in_usd'].mean()
print(average_hourly_rate_by_size)

# Create box plot
sns.boxplot(x='company_size', y='average_hourly_rate__per_hr_in_usd', data=df)
plt.show()










# Converting the columns.


df['minimum_project_size_in_usd'] = pd.to_numeric(df['minimum_project_size_in_usd'],errors='coerce')
print(df)

# Now we will check how many null values we have after the conversion to numeric
print(df.isnull().sum())

# So we will handle NaN values to avoid null output

df.fillna({'number_of_employees': 249} , inplace=True)
df.fillna({'minimum_project_size_in_usd': 1000} , inplace=True)


# Calculating the project complexity.

def calculate_complexity(row):
    # Normalize values (adjust weights as needed)
    normalized_employees = row['number_of_employees'] / max(df['number_of_employees'])
    normalized_hourly_rate = row['average_hourly_rate__per_hr_in_usd'] / max(df['average_hourly_rate__per_hr_in_usd'])
    normalized_project_size = row['minimum_project_size_in_usd'] / max(df['minimum_project_size_in_usd'])

    # Calculate complexity score (adjust weights as needed)
    complexity_score = normalized_employees * 0.3 + normalized_hourly_rate * 0.3 +  normalized_project_size * 0.4

    return complexity_score

df['project_complexity'] = df.apply(calculate_complexity, axis=1)


print(df['project_complexity'])  # Print the entire column











# Calculate correlation coefficient
correlation = df['project_complexity'].corr(df['average_hourly_rate__per_hr_in_usd'])
print("Correlation coefficient:", correlation)


# Group analysis
average_hourly_rate_by_complexity = df.groupby('project_complexity')['average_hourly_rate__per_hr_in_usd'].mean()
print(average_hourly_rate_by_complexity)

# Analyzing project complexity and hourly rates

sns.scatterplot(x='project_complexity', y='average_hourly_rate__per_hr_in_usd', hue='location_category', data=df)
plt.title('Hourly Rate vs. Project Complexity by Location')
plt.xlabel('Project Complexity')
plt.ylabel('Average Hourly Rate')
plt.show()



# Drop rows with missing values in X or y
df_cleaned = df.dropna(subset=['project_complexity', 'average_hourly_rate__per_hr_in_usd'])

# Update X and y with the cleaned DataFrame
X = df_cleaned[['project_complexity']]
y = df_cleaned['average_hourly_rate__per_hr_in_usd']

model = sm.OLS(y, X).fit()
print(model.summary())















