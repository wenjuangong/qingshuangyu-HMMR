import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap='viridis')
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
data_path = 'ensemble.txt'
t = []
p = []
truth = []
pred = []
with open(data_path) as f:
    data = f.readline()
    while data:
        t.append(int(data[0]))
        p.append(int(data[2]))
        data = f.readline()

print("test original label: ",t)
print('\n')
print("test predict label: ", p)
print('\n')

label_dict = {0:'Cha-cha',1:'Rumba',2:'Tango',3:'Waltz'}
for i in range(len(t)):
    truth.append(label_dict[t[i]])
    pred.append(label_dict[p[i]])
#labels表示你不同类别的代号，比如这里的demo中有13个类别
labels = ['Cha-cha','Rumba','Tango','Waltz']
tick_marks = np.array(range(len(labels))) + 0.5

cm = confusion_matrix(truth, pred)
df_cm = pd.DataFrame(cm, index = ['Cha-cha','Rumba','Tango','Waltz'], columns = ['Cha-cha','Rumba','Tango','Waltz'])
sns.set()
f,ax=plt.subplots()

sns.heatmap(df_cm,annot=False,ax=ax)#画热力图

ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴

# np.set_printoptions(precision=2)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print(cm_normalized)
# plt.figure(figsize=(12, 8), dpi=120)
#
# ind_array = np.arange(len(labels))
# x, y = np.meshgrid(ind_array, ind_array)
#
# for x_val, y_val in zip(x.flatten(), y.flatten()):
#     c = cm_normalized[y_val][x_val]
#     if c > 0.01:
#         plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# # offset the tick
# plt.gca().set_xticks(tick_marks, minor=True)
# plt.gca().set_yticks(tick_marks, minor=True)
# plt.gca().xaxis.set_ticks_position('none')
# plt.gca().yaxis.set_ticks_position('none')
# plt.grid(True, which='minor', linestyle='-')
# plt.gcf().subplots_adjust(bottom=0.15)
#
# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# # show confusion matrix
plt.savefig('./data/confusion_matrix.png', format='png')
plt.show()
#
#
#
