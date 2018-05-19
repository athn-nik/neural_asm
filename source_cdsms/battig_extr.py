import pickle

with open('/home/n_athan/Desktop/diploma/data/task_categorization/Battig/word_catagor.pkl', 'rb') as unt_tok:
 w_cat = pickle.load(unt_tok)
x=w_cat.keys()
for c in x:
 print(c)

