import machineLearning.tab.tab as tab

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.metrics import precision_recall_fscore_support as s
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np
import sys
import random
import os
import shutil
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


def createDirectory(path, folderName):
    final = os.path.join(path, folderName)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.makedirs(final)


def isNumber(s):
    try:
        return float(s)
    except ValueError:
        return s


def getTruePositive(matrix, class_index):
    return matrix[class_index][class_index]


def getFalseNegative(matrix, class_index):
    return sum(matrix[class_index]) - matrix[class_index][class_index]


def getTrueNegative(matrix, class_index):
    tmp = matrix
    tmp = np.delete(tmp, class_index, axis=1)
    tmp = np.delete(tmp, class_index, axis=0)
    return sum(sum(tmp))


def getFalsePositive(matrix, class_index):
    tmp = matrix
    tmp = tmp[:, class_index]
    tmp = np.delete(tmp, class_index, axis=0)
    return sum(tmp)


def getSupport(matrix, class_index):
    return sum(matrix[class_index])


def getTotalTruePositive(matrix):
    sum1 = 0
    for i in range(len(matrix)):
        sum1 = sum1 + matrix[i][i]
    return sum1


def getTotalTrueNegative(matrix):
    sum1 = 0
    for i in range(len(matrix)):
        sum1 = sum1 + getTrueNegative(matrix, i)
    return sum1


def getTotalFalsePositive(matrix):
    sum1 = 0
    for i in range(len(matrix)):
        sum1 = sum1 + getFalsePositive(matrix, i)
    return sum1


def getTotalFalseNegative(matrix):
    sum1 = 0
    for i in range(len(matrix)):
        sum1 = sum1 + getFalseNegative(matrix, i)
    return sum1


def _createDummies(data, dummiesList):
    list1 = []
    for col in dummiesList:
        if col in data.columns:
            list1.append(col)
    data = pd.get_dummies(data, columns=list1)
    return data, list1


def _normalize(data, target):
    scaler = MinMaxScaler()
    columns = data.drop(target, axis=1).columns
    data[columns] = scaler.fit_transform(data[columns])
    return data


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_indexes_max_value(l):
    max_value = max(l)
    if l.count(max_value) > 1:
        return [i for i, x in enumerate(l) if x == max(l)], float(max_value)
    else:
        return l.index(max(l)), float(max_value)


def get_indexes_min_value(l):
    min_value = min(l)
    if l.count(min_value) > 1:
        return [i for i, x in enumerate(l) if x == min(l)], float(min_value)
    else:
        return l.index(min(l)), float(min_value)


def best(self, lst2, Flst):
    u, c = np.unique(self.data[self.target], return_counts=True)
    unique2 = list(u)
    chunks = [Flst[x:x + (len(unique2) + 1)] for x in range(0, len(Flst), (len(unique2) + 1))]
    Ftab = []
    v2 = ''
    for i in range(len(unique2) + 1):
        best = [row[i] for row in chunks]
        tab = []
        if i < len(unique2):
            tab.append(unique2[i])
        else:
            tab.append('Total')
        for j in range(2, len(Flst[0])):
            if j == 4 or j == 5:
                tmp, tmpv = get_indexes_min_value([row[j] for row in best])
            else:
                tmp, tmpv = get_indexes_max_value([row[j] for row in best])
            if not isinstance(tmp, int):
                for k in range(len(tmp)):
                    if k > 0:
                        v2 = v2 + ',' + lst2[tmp[k]]
                    else:
                        v2 = lst2[tmp[k]]
            else:
                v2 = lst2[tmp]
            if tmpv.is_integer():
                tab.append(v2 + ' (' + str(int("{:.0f}".format(tmpv))) + ')')
            else:
                tab.append(v2 + ' (' + str(float("{:.2f}".format(tmpv))) + ')')
        Ftab.extend([tab[:]])
    return Ftab


def ready(self, ind, data, dummiesList, createDummies, normalize):
    if createDummies:
        data, dummiesLst = _createDummies(data, dummiesList)
    tmp = data.drop([self.target], axis=1)
    tmp2 = data[self.target]
    data = tmp.iloc[:, ind]
    cols = data.columns
    kwargs = {self.target: lambda x: pd.Series(tmp2).values}
    data = data.assign(**kwargs)
    data = data.dropna()
    if normalize:
        data = _normalize(data, self.target)
    X = data.drop([self.target], axis=1).values
    y = data[self.target].values
    return X, y, cols


def list_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [idx for idx, item in enumerate(seq) if item in seen or seen_add(item)]


def create_population(inds, size):
    pop = np.zeros((inds, size), dtype=bool)
    for i in range(inds):
        pop[i, 0:random.randint(0, size)] = True
        np.random.shuffle(pop[i])
    return pop


def fitness(self, pop, mode, data, dummiesList, createDummies, normalize, metric):
    scores = []
    models = []
    inds = []
    colsLst = []
    scoresA = []
    scoresP = []
    scoresR = []
    scoresF = []

    matrix_length = len(np.unique(self.data[self.target]))

    if mode == 'svm':
        model = LinearSVC(class_weight='balanced', random_state=1)
    elif mode == 'rdc':
        model = RandomForestClassifier(n_estimators=30, bootstrap=False, class_weight='balanced', random_state=1)
    elif mode == 'dtc':
        model = DecisionTreeClassifier(class_weight='balanced', random_state=1)
    elif mode == 'etc':
        model = ExtraTreesClassifier(class_weight='balanced', random_state=1)
    elif mode == 'lda':
        model = LinearDiscriminantAnalysis()
    elif mode == 'gnb':
        model = GaussianNB()
    elif mode == 'rrc':
        model = RidgeClassifier(class_weight='balanced')
    else:
        model = LogisticRegression(solver='liblinear', C=10.0, class_weight='balanced')
    k = model_selection.StratifiedKFold(5)
    for ind in pop:
        if not any(ind):
            ind[random.randint(0, len(ind) - 1)] = True
        try:
            tab_data, tab_val = tab.get([int(x) for x in ind], self.tab_data, self.tab_vals)
            tab_val = np.array(tab_val)
            accuracy = (getTotalTruePositive(tab_val) + getTotalTrueNegative(tab_val)) / \
                       (getTotalTruePositive(tab_val) + getTotalTrueNegative(tab_val) +
                        getTotalFalsePositive(tab_val) + getTotalFalseNegative(tab_val))
            precision_tab = []
            recall_tab = []
            for i in range(len(tab_val)):
                a = getTruePositive(tab_val, i) / (getFalsePositive(tab_val, i) + getTruePositive(tab_val, i))
                b = getTruePositive(tab_val, i) / (getFalseNegative(tab_val, i) + getTruePositive(tab_val, i))
                precision_tab.append(a)
                recall_tab.append(b)
            precision = sum(precision_tab)/len(precision_tab)
            recall = sum(recall_tab)/len(recall_tab)
            fscore = 2 * (1 / ((1 / precision) + (1 / recall)))
            matrix = tab_val
            tmp = self.data.drop([self.target], axis=1)
            tmp = tmp.iloc[:, ind]
            cols = tmp.columns
            self.tab_find = self.tab_find + 1
        except:
            matrix = np.zeros((matrix_length, matrix_length), dtype=int)
            X, y, cols = ready(self, ind, data, dummiesList, createDummies, normalize)
            originalclass = []
            predictedclass = []
            for train_index, test_index in k.split(X, y):  # Split in X
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if mode == ('knn' or 'gnb' or 'lda'):
                    if mode == 'knn':
                        model = KNeighborsClassifier(n_neighbors=int(len(X_train) ** (1 / 2)))
                    sm = SMOTE(sampling_strategy='auto')
                    X_train, y_train = sm.fit_resample(X_train, y_train)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                originalclass.extend(y_test)
                predictedclass.extend(y_pred)

                matrix += confusion_matrix(y_test, y_pred)

            accuracy = (getTotalTruePositive(matrix) + getTotalTrueNegative(matrix)) / \
                       (getTotalTruePositive(matrix) + getTotalTrueNegative(matrix) +
                        getTotalFalsePositive(matrix) + getTotalFalseNegative(matrix))
            precision, recall, fscore, support = s(originalclass, predictedclass, average='macro')
            self.tab_data, self.tab_vals = tab.add([int(x) for x in ind], matrix.tolist(), self.tab_data,
                                                   self.tab_vals)
            self.tab_insert = self.tab_insert + 1

        models.append(matrix)
        scoresA.append(accuracy)
        scoresP.append(precision)
        scoresR.append(recall)
        scoresF.append(fscore)
        inds.append(ind)
        colsLst.append(cols)
        if metric == 'accuracy' or metric == 'exactitude':
            scores.append(accuracy)
        elif metric == 'recall' or metric == 'rappel':
            scores.append(recall)
        elif metric == 'precision' or metric == 'précision':
            scores.append(precision)
        elif metric == 'fscore':
            scores.append(fscore)
        else:
            scores.append(accuracy)

    return scores, models, inds, colsLst, scoresA, scoresP, scoresR, scoresF, self


def fitness2(self, mode, solution, data, dummiesList, createDummies, normalize):

    matrix_length = len(np.unique(self.data[self.target]))

    if mode == 'svm':
        model = LinearSVC(class_weight='balanced', random_state=1)
    elif mode == 'rdc':
        model = RandomForestClassifier(n_estimators=30, bootstrap=False, class_weight='balanced', random_state=1)
    elif mode == 'dtc':
        model = DecisionTreeClassifier(class_weight='balanced', random_state=1)
    elif mode == 'etc':
        model = ExtraTreesClassifier(class_weight='balanced', random_state=1)
    elif mode == 'lda':
        model = LinearDiscriminantAnalysis()
    elif mode == 'gnb':
        model = GaussianNB()
    elif mode == 'rrc':
        model = RidgeClassifier(class_weight='balanced')
    else:
        model = LogisticRegression(solver='liblinear', C=10.0, class_weight='balanced')
    k = model_selection.StratifiedKFold(5)
    if not any(solution):
        solution[random.randint(0, len(solution) - 1)] = True
    try:
        tab_data, tab_val = tab.get([int(x) for x in solution], self.tab_data, self.tab_vals)
        tab_val = np.array(tab_val)
        accuracy = (getTotalTruePositive(tab_val) + getTotalTrueNegative(tab_val)) / \
                   (getTotalTruePositive(tab_val) + getTotalTrueNegative(tab_val) +
                    getTotalFalsePositive(tab_val) + getTotalFalseNegative(tab_val))
        precision_tab = []
        recall_tab = []
        for i in range(len(tab_val)):
            a = getTruePositive(tab_val, i) / (getFalsePositive(tab_val, i) + getTruePositive(tab_val, i))
            b = getTruePositive(tab_val, i) / (getFalseNegative(tab_val, i) + getTruePositive(tab_val, i))
            precision_tab.append(a)
            recall_tab.append(b)
        precision = sum(precision_tab)/len(precision_tab)
        recall = sum(recall_tab)/len(recall_tab)
        fscore = 2 * (1 / ((1 / precision) + (1 / recall)))
        matrix = tab_val
        tmp = self.data.drop([self.target], axis=1)
        tmp = tmp.iloc[:, solution]
        cols = tmp.columns
        self.tab_find = self.tab_find + 1
    except:
        matrix = np.zeros((matrix_length, matrix_length), dtype=int)
        X, y, cols = ready(self, solution, data, dummiesList, createDummies, normalize)
        originalclass = []
        predictedclass = []
        for train_index, test_index in k.split(X, y):  # Split in X
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if mode == ('knn' or 'lda' or 'gnb'):
                if mode == 'knn':
                    model = KNeighborsClassifier(n_neighbors=int(len(X_train) ** (1 / 2)))
                sm = SMOTE(sampling_strategy='auto')
                X_train, y_train = sm.fit_resample(X_train, y_train)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            originalclass.extend(y_test)
            predictedclass.extend(y_pred)

            matrix += confusion_matrix(y_test, y_pred)

        accuracy = (getTotalTruePositive(matrix) + getTotalTrueNegative(matrix)) / \
                   (getTotalTruePositive(matrix) + getTotalTrueNegative(matrix) +
                    getTotalFalsePositive(matrix) + getTotalFalseNegative(matrix))
        precision, recall, fscore, support = s(originalclass, predictedclass, average='macro')
        self.tab_data, self.tab_vals = tab.add([int(x) for x in solution], matrix.tolist(), self.tab_data,
                                               self.tab_vals)
        self.tab_insert = self.tab_insert + 1

    return accuracy, recall, precision, fscore, cols, matrix, self


def getList(bestModel, bestScore, bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestCols, indMax, unique, mode):
    list1 = []
    flist = []
    specificity_tab = []

    list1.append(list(bestCols))
    flist.append(list(bestCols))

    for i in range(len(bestModel)):
        list2 = []
        a = getTruePositive(bestModel, i) / (getFalsePositive(bestModel, i) + getTruePositive(bestModel, i))
        b = getTruePositive(bestModel, i) / (getFalseNegative(bestModel, i) + getTruePositive(bestModel, i))
        c = getTrueNegative(bestModel, i) / (getFalsePositive(bestModel, i) + getTrueNegative(bestModel, i))
        d = (getTruePositive(bestModel, i) + getTrueNegative(bestModel, i)) / \
            (getTruePositive(bestModel, i) + getTrueNegative(bestModel, i) +
             getFalsePositive(bestModel, i) + getFalseNegative(bestModel, i))
        list2.extend((unique[i], getSupport(bestModel, i), getTruePositive(bestModel, i),
                      getTrueNegative(bestModel, i), getFalsePositive(bestModel, i),
                      getFalseNegative(bestModel, i), d,
                      a, c, b, (a + b) / 2, 2 * (1 / ((1 / a) + (1 / b))),
                      (1 + (2 ** 2)) * (a * b) / ((2 ** 2) * a + b)))
        list1.append(list2[:])
        flist.append(list2[:])
        specificity_tab.append(c)
    specificity = sum(specificity_tab)/len(specificity_tab)
    tmp = ['Total', sum(sum(bestModel)), getTotalTruePositive(bestModel),
           getTotalTrueNegative(bestModel), getTotalFalsePositive(bestModel),
           getTotalFalseNegative(bestModel),
           bestScoreA, bestScoreP, specificity, bestScoreR, (bestScoreP + bestScoreR) / 2, bestScoreF,
           (1 + (2 ** 2)) * (bestScoreP * bestScoreR) / ((2 ** 2) * bestScoreP + bestScoreR)]
    list1.append(tmp[:])
    flist.append(tmp[:])
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            if isinstance(list1[i][j], float):
                if j != 0:
                    list1[i][j] = float("{:.2f}".format(list1[i][j] * 100))
                    flist[i][j] = str(float("{:.2f}".format(flist[i][j] * 100))) + '%'
    return list1, flist


def res(heuristic, x, y, z, besties, names, iters, times, names2, metric, path, n_gen, self):
    folderName = "Ulti"
    createDirectory(path, folderName)
    cmap = ['dodgerblue', 'red', 'springgreen', 'gold', 'orange', 'deeppink', 'darkviolet', 'blue', 'dimgray',
            'salmon', 'green', 'cyan', 'indigo', 'crimson', 'chocolate', 'black']
    fig, ax = plt.subplots()
    i = 0
    for val in besties:
        ax.plot(list(range(0, len(val))), val, color=cmap[i])
        i = i + 1
    ax.set_title("Evolution du score par génération"
                 + "\n" + heuristic + "\n" + self.data_name)
    ax.set_xlabel("génération")
    ax.set_ylabel(metric)
    ax.grid()
    ax.legend(labels=names,
              loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plot_' + '.png')
    b = os.path.join(os.getcwd(), a)
    fig.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    i = 0
    for val in times:
        ax2.plot(list(range(0, len(val))), val, color=cmap[i])
        i = i + 1
    ax2.set_title("Evolution du temps par génération"
                 + "\n" + heuristic + "\n" + self.data_name)
    ax2.set_xlabel("génération")
    ax2.set_ylabel("Temps en seconde")
    ax2.grid()
    ax2.legend(labels=names2,
               loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    a = os.path.join(os.path.join(path, folderName), 'plotTps_' + '.png')
    b = os.path.join(os.getcwd(), a)
    fig2.savefig(os.path.abspath(b), bbox_inches="tight")
    plt.close(fig2)

    for i in range(len(x)):
        z.append(x[i][1:])
    z = np.reshape(z, (-1, len(z[0][0])))
    z = z.tolist()
    for i in range(len(z)):
        for j in range(len(z[i])):
            z[i][j] = isNumber(z[i][j])

    """
    x contient pour chaque modèle la meilleure solution (le résultat pour chaque classe et les variables)
    y même chose mais chque chiffre est en pourcentage
    z même chose mais le nombre de dimension a été réduit de 1 et les variables sélectionnées sont enlevées
    """

    # print(x[0][-1][9])
    maxi_score = 0
    maxi_col = 0
    if metric == 'accuracy' or metric == 'exactitude':
        index_ = 6
    elif metric == 'recall' or metric == 'rappel':
        index_ = 9
    elif metric == 'precision' or metric == 'précision':
        index_ = 7
    elif metric == 'fscore':
        index_ = 11
    else:
        index_ = 6
    for val in x:
        if val[-1][index_] > maxi_score:
            maxi_score = val[-1][index_]
            maxi_col = val[0]

    Ftab = best(self, f7(names), z)
    print("Meilleur")
    print()
    print(Ftab)
    print()
    print()

    a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
    f = open(a, "a")
    f.write("x: " + str(x) + os.linesep)
    f.write("y: " + str(y) + os.linesep)
    f.write("z: " + str(z) + os.linesep)
    f.write("ftab: " + str(Ftab) + os.linesep)
    f.write("score max: " + str(maxi_score) + os.linesep)
    f.write("colonne max:" + str(list(maxi_col)) + os.linesep)
    f.close()

    return x, y, z, [maxi_score, maxi_col], Ftab