#importing pandas as pd
import pandas as pd
  
name = 'pessoas'

# Read and store content
# of an excel file 
read_file = pd.read_excel (name + ".xlsx")
  
# Write the dataframe object
# into csv file
read_file.to_csv (name + ".csv", 
                  index = None,
                  header=True, encoding='utf-8')
    