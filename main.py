import pandas as pd

#Creating DataFrame

df = pd.read_csv('AI_Companies.csv')
#print(df)

# Set the display option to show all columns

pd.set_option('display.max_columns' , None)
print(df)