
import pandas as pd
conda install mlxtend #installing mlxextend
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("C:/Users/user/Desktop/DATASETS/book.csv") #loading the dataset
df.columns
df.count
df.info
df.describe

frequent_itemsets = apriori(df, min_support = 0.0085, max_len = 5, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

# barplot of top 10 
import matplotlib.pyplot as plt
plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=10)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1) #writing rules by arules
rules.head(15)
rules.sort_values('lift', ascending = False).head(10)  #sorting values using highest lift ratio

#considering only required columns and removing profusion
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list) #considering antecedent and consequent columns and coverting them to list

ma_X = ma_X.apply(sorted) #sorting in sequence

rules_sets = list(ma_X) #coverting from str to list

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)] #using set to remove profusion(duplicates)

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
highest_lift_ratio = rules_no_redudancy.sort_values('lift', ascending = False).head(10)
