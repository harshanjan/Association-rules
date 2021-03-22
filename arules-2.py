
conda install mlxtend  #to import packages like apriori, association rules 
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
#the data is in string format and we cannot import(read) directly in python so creating a function and loading it.
groceries = []
with open("C:/Users/user/Desktop/DATASETS/groceries.csv") as f:
    groceries = f.read()

# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

groceries_list = [] 
for i in groceries:
    groceries_list.append(i.split(","))  #splitting each string in a row separately

all_groceries_list = [i for item in groceries_list for i in item]

from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_groceries_list) #we get the count of each item

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])   #lambda is anonymous function which can be written in single line

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'red')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


groceries_series = pd.DataFrame(pd.Series(groceries_list)) # Creating Data Frame for the transactions data
groceries_series = groceries_series.iloc[:9835, :] # removing the last empty transaction

groceries_series.columns = ["transactions"]  #changing column name as transactions

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True) # using apriori with required parameters

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=10)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

defining a function # for sorting, converting to list and removing profusion
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list) #considering only required columns

ma_X = ma_X.apply(sorted) #sorting values

rules_sets = list(ma_X) #converting to list

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)] #removing profusion

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

rules_no_redudancy.sort_values('lift', ascending = False).head(10) # Sorting them with respect to list and getting top 10 rules 
