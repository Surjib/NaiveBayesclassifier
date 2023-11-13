import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction


#read data from .csv
data_frame = pd.read_csv('D:/DZ/11sem/AI_Enregy/LR2/spam_test.csv',  encoding = "ISO-8859-1")
print(data_frame);


# delete obsolete columns
data_frame.drop(["Unnamed: 2",  "Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1, inplace = True)
print(data_frame);


#draw pie chart for target column
target = pd.value_counts(data_frame['v1'])
target.plot(kind = 'pie', autopct='%1.1f%%')
plt.title('Target variable values')
plt.ylabel('')

plt.show()


#return dataframe [words, count(freq)] in non-spam mess
ham_words = Counter("".join(data_frame[data_frame['v1']=='ham']['v2']). \
split()).most_common(20)
df_ham_words = pd.DataFrame.from_dict(ham_words)
df_ham_words = df_ham_words.rename(columns={0: 'words in non-spam', 1:'count'})

#return dataframe [words, count(freq)] in spam mess
spam_words = Counter("".join(data_frame[data_frame['v1']=='spam']['v2']). \
split()).most_common(20)
df_spam_words = pd.DataFrame.from_dict(spam_words)
df_spam_words = df_spam_words.rename(columns={0: 'words in spam', 1:'count'})




# plt.subplot(2, 1, 1)
df_ham_words.plot.bar(legend = False, subplots = True)
y_pos = np.arange(len(df_ham_words['words in non-spam']))
plt.xticks(y_pos, df_ham_words['words in non-spam'])
plt.title('more frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('count')

# plt.subplot(2, 1, 2)
df_spam_words.plot.bar(legend = False)
y_pos = np.arange(len(df_spam_words['words in spam']))
plt.xticks(y_pos, df_spam_words['words in spam'])
plt.title('more frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('count')

plt.show()



tokenizer = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = tokenizer.fit_transform(data_frame['v2'])

print(X)
