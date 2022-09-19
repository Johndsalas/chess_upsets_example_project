# Prunes Object Columns to those with more than 50 instances 
# columns = ['opening_name', 'time_code', 'opening_code']

# for column in columns:
    
#     value_set = set(df[column].to_list())
    
#     value_set_above_50 = [value for value in value_set if df[column].value_counts()[value] >= 50]
    
#     print(column.upper())
#     print(value_set_above_50)
#     print('Length:', len(value_set_above_50))
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     print()
    
#     df[column] = df[column].apply(lambda value : value if value in value_set_above_50 else 'Unpopuler')
    
# print("VALUE COUNTS")
# for column in columns:
    
#     print(column.upper())
#     print(df[column].value_counts())
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     print()


# print("VALUE COUNTS")
# for column in columns:
    
#     print(column.upper())
#     print(df[column].value_counts())
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     print()


# removing rows containing object values that appear less than 30 times
        for column in list(df.select_dtypes(include=['object']).columns):
        
            value_set = set(df[column].to_list())
        
            value_set_above_30 = [value for value in value_set if df[column].value_counts()[value] >= 30]

            df[column] = df[column].apply(lambda value : value if value in value_set_above_30 else np.nan)

        #df = df.dropna()


list(df.select_dtypes(include=['object']).columns)