import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def find_optimal_param(param_range, X_train, X_test, Y_train, Y_test, isMultiNB):
    train_accuracy = []
    test_accuracy = []
    test_recall = []
    test_precision = []
    param = ''
    for k in param_range:
        if isMultiNB:
            model = MultinomialNB(alpha=k)  # создание модели с заданной метрикой
            param = 'alpha'
        else:
            model = SVC(C = k)
            param = 'C'

        model.fit(X_train, Y_train)  # Обучение модели
        # print("our accuracy is:{}".format(multNB.score(X_train, Y_train)))

        Y_predict = model.predict(X_test)
        Y_predict_train = model.predict(X_train)

        train_accuracy.append(metrics.accuracy_score(Y_train, Y_predict_train))
        test_accuracy.append(metrics.accuracy_score(Y_test, Y_predict))
        test_recall.append(metrics.recall_score(Y_test, Y_predict))
        test_precision.append(metrics.precision_score(Y_test, Y_predict))

    matrix = np.matrix(np.c_[param_range, train_accuracy, test_accuracy, test_recall, test_precision])
    models = pd.DataFrame(data=matrix,
                          columns=[param, 'train accuracy', 'test accuracy', 'test recall', 'test precision'])

    best_index = models['test precision'].idxmax()
    best_value = models.iloc[best_index]['test precision']
    best_index = models[models['test precision'] == best_value]['test accuracy'].idxmax()
    best_value = param_range[best_index]

    print('Best precision and accuracy using alpha value {} index {} '.format(best_value, best_index))
    print('--------------------------')

    return best_value, best_index, train_accuracy, test_accuracy, test_recall, test_precision


def plot_score(best_value, best_index, train_accuracy, test_accuracy, test_recall, test_precision, param_range):
    plt.subplot(2, 2, 1)
    plt.plot(param_range, train_accuracy)
    plt.plot(best_value, train_accuracy[best_index], marker='o', color='red')
    plt.annotate(xy=(best_value, train_accuracy[best_index]),
                 text='({}, {})'.format(("%.1f" % best_value), ("%.3f" % train_accuracy[best_index])))
    plt.title("Train Accuracy", fontsize=10)
    plt.ylabel('Accuracy score(%)', fontsize=8)
    plt.xlabel('alpha value', fontsize=8)
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(param_range, test_accuracy)
    plt.plot(best_value, test_accuracy[best_index], marker='o', color='red')
    plt.annotate(xy=(best_value, test_accuracy[best_index]),
                 text='({}, {})'.format(("%.1f" % best_value), ("%.3f" % test_accuracy[best_index])))
    plt.title("Test Accuracy", fontsize=10)
    plt.ylabel('Accuracy score(%)', fontsize=8)
    plt.xlabel('alpha value', fontsize=8)
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(param_range, test_recall)
    plt.plot(best_value, test_recall[best_index], marker='o', color='red')
    plt.annotate(xy=(best_value, test_recall[best_index]),
                 text='({}, {})'.format(("%.1f" % best_value), ("%.3f" % test_recall[best_index])))
    plt.title("Test Recall", fontsize=10)
    plt.ylabel('Recall score(%)', fontsize=8)
    plt.xlabel('alpha value', fontsize=8)
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(param_range, test_precision)
    plt.plot(best_value, test_precision[best_index], marker='o', color='red')
    plt.annotate(xy=(best_value, test_precision[best_index]),
                 text='({}, {})'.format(("%.1f" % best_value), ("%.3f" % test_precision[best_index])))
    plt.title("Test Precision", fontsize=10)
    plt.ylabel('recall score(%)', fontsize=8)
    plt.xlabel('alpha value', fontsize=8)
    plt.grid(True)

    plt.show()

#read data from .csv
data_frame = pd.read_csv('D:/DZ/11sem/AI_Enregy/LR2/spam_test.csv',  encoding = "ISO-8859-1")
print(data_frame)


print('--------------------------')

# delete obsolete columns
data_frame.drop(["Unnamed: 2",  "Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1, inplace = True)
print(data_frame)

print('--------------------------')

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

X = tokenizer.fit_transform(data_frame['v2']) # матрица количества токенов

tokens = tokenizer.get_feature_names_out()  #8404 отдельных токена (N грамм)
# print(len(tokens))

rep = X.toarray()
# print(rep)

# print('-')




data_frame['v1'] = data_frame['v1'].map({'spam': 1, 'ham': 0})# замена текстовой информации целового столбца на 1 и 0 (категориальный признак в числовой)

print(data_frame['v1'])
print('--------------------------')


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, data_frame['v1'], test_size = 0.33)


alpha_range = np.arange(0.1, 20, 0.1)

# Поиск оптимального параметра alpha по метрикам accuracy и precision
best_value, best_index, train_accuracy, test_accuracy, test_recall, test_precision = find_optimal_param(alpha_range, X_train, X_test, Y_train, Y_test, True)


optimal_multNB = MultinomialNB(alpha = alpha_range[best_index])
optimal_multNB.fit(X_train, Y_train)


plot_score(best_value, best_index, train_accuracy, test_accuracy, test_recall, test_precision, alpha_range)

confusion_matrix_NB = confusion_matrix(Y_test, optimal_multNB.predict(X_test))
conf_matrix = pd.DataFrame(data = confusion_matrix_NB, columns = ['predicted ham', 'predicted spam'], index = ['actual ham', 'actual spam'])

print(conf_matrix)
print('--------------------------')


Y_pred_pr = optimal_multNB.predict_proba(X_test)[:,1]
fpr, tpr, threshold = metrics. roc_curve (Y_test, Y_pred_pr)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic for ')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid('on')
plt.show()


C_range = np.arange(0.01, 3, 0.1)

# Поиск оптимального параметра C по метрикам accuracy и precision
best_value_svc, best_index_svc, train_accuracy_svc, test_accuracy_svc, test_recall_svc, test_precision_svc = find_optimal_param(C_range, X_train, X_test, Y_train, Y_test, False)


optimal_SVC = SVC(C = C_range[best_index_svc], probability= True)
optimal_SVC.fit(X_train, Y_train)


plot_score(best_value_svc, best_index_svc, train_accuracy_svc, test_accuracy_svc, test_recall_svc, test_precision_svc, C_range)

confusion_matrix_SVC = confusion_matrix(Y_test, optimal_SVC.predict(X_test))
conf_matrix_SVC = pd.DataFrame(data = confusion_matrix_SVC, columns = ['predicted ham', 'predicted spam'], index = ['actual ham', 'actual spam'])

print(conf_matrix_SVC)
print('--------------------------')


Y_pred_pr = optimal_SVC.predict_proba(X_test)[:,1]
fpr_SVC, tpr_SVC, threshold_SVC = metrics.roc_curve (Y_test, Y_pred_pr)

roc_auc = metrics.auc(fpr_SVC, tpr_SVC)

plt.title('Receiver Operating Characteristic for SVC')
plt.plot(fpr_SVC, tpr_SVC, 'b', label = 'AUC = %0.2f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid('on')
plt.show()

