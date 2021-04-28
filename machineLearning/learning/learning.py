import machineLearning.preprocessing.data as data

import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support as s
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class Learning:

    """Classe définissant la partie d'aprentissage pour un modèle"""

    def __init__(self, data, listModels, target, copy):
        self.data = data
        self.listModels = listModels
        self.target = target
        self.copy = copy

    @staticmethod
    def __classification_report_with_accuracy_score(y_true, y_pred, originalclass, predictedclass):
        originalclass.extend(y_true)
        predictedclass.extend(y_pred)

    @staticmethod
    def __getTruePositive(matrix, class_index):
        return matrix[class_index][class_index]

    @staticmethod
    def __getFalseNegative(matrix, class_index):
        return sum(matrix[class_index]) - matrix[class_index][class_index]

    @staticmethod
    def __getTrueNegative(matrix, class_index):
        tmp = matrix
        tmp = np.delete(tmp, class_index, axis=1)
        tmp = np.delete(tmp, class_index, axis=0)
        return sum(sum(tmp))

    @staticmethod
    def __getFalsePositive(matrix, class_index):
        tmp = matrix
        tmp = tmp[:, class_index]
        tmp = np.delete(tmp, class_index, axis=0)
        return sum(tmp)

    @staticmethod
    def __getSupport(matrix, class_index):
        return sum(matrix[class_index])

    @staticmethod
    def __getTotalTruePositive(matrix):
        sum1 = 0
        for i in range(len(matrix)):
            sum1 = sum1 + matrix[i][i]
        return sum1

    def __getTotalTrueNegative(self, matrix):
        sum1 = 0
        for i in range(len(matrix)):
            sum1 = sum1 + self.__getTrueNegative(matrix, i)
        return sum1

    def __getTotalFalsePositive(self, matrix):
        sum1 = 0
        for i in range(len(matrix)):
            sum1 = sum1 + self.__getFalsePositive(matrix, i)
        return sum1

    def __getTotalFalseNegative(self, matrix):
        sum1 = 0
        for i in range(len(matrix)):
            sum1 = sum1 + self.__getFalseNegative(matrix, i)
        return sum1

    def f7(self, seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def get_indexes_max_value(self, l):
        max_value = max(l)
        # print(max_value)
        if l.count(max_value) > 1:
            return [i for i, x in enumerate(l) if x == max(l)], float(max_value)
        else:
            return l.index(max(l)), float(max_value)

    def get_indexes_min_value(self, l):
        min_value = min(l)
        # print(min_value)
        if l.count(min_value) > 1:
            return [i for i, x in enumerate(l) if x == min(l)], float(min_value)
        else:
            return l.index(min(l)), float(min_value)

    def best(self, lst2, Flst):
        u, c = np.unique(self.copy[self.target], return_counts=True)
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
                    tmp, tmpv = self.get_indexes_min_value([row[j] for row in best])
                else:
                    tmp, tmpv = self.get_indexes_max_value([row[j] for row in best])
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

    def init(self):
        print("###############")
        print("#APPRENTISSAGE#")
        print("###############")
        print()

        matrix_length = len(np.unique(self.data[self.target]))
        X = self.data.drop([self.target], axis=1).values
        y = self.data[self.target].values

        u, c = np.unique(self.copy[self.target], return_counts=True)
        u = list(u)

        y_bin = label_binarize(y, classes=u)
        n_classes = y_bin.shape[1]

        lst = []
        Flist = []
        mNameList = []
        mNameList2 = []

        if isinstance(self.listModels, str):
            if self.listModels == 'all':
                self.listModels = ['x', 'rrc', 'sgd', 'knn', 'svm', 'rbf', 'dtc', 'rdc', 'etc', 'gbc', 'abc', 'bac',
                                   'lda', 'qda', 'gnb']
            else:
                self.listModels = ['x']

        for mods in self.listModels:
            matrix = np.zeros((matrix_length, matrix_length), dtype=int)

            model = LogisticRegression(solver='liblinear', C=10.0, class_weight='balanced')

            if mods == 'sgd':
                mName = 'Descente du gradient stochastique [SGD]'
                mName2 = 'SGD'
                model = SGDClassifier(class_weight='balanced', loss='modified_huber')
                print("SGDClassifier\n")
            elif mods == 'knn':
                mName = 'K plus proches voisins [KNN]'
                mName2 = 'KNN'
                print("KNeighborsClassifier\n")
            elif mods == 'svm':
                mName = 'Machine à vecteurs de support (linéaire) [SVM]'
                mName2 = 'SVM'
                model = SVC(kernel='linear', class_weight='balanced', probability=True)
                print("SVM\n")
            elif mods == 'rbf':
                mName = 'Machine à vecteurs de support (fonction de base radiale) [RBF]'
                mName2 = 'RBF'
                model = SVC(kernel='rbf', class_weight='balanced', probability=True)
                print("SVM rbf\n")
            elif mods == 'pol':
                mName = 'Machine à vecteurs de support polynomial [POL]'
                mName2 = 'POL'
                model = SVC(kernel='poly', class_weight='balanced', probability=True)
                print("SVM poly\n")
            elif mods == 'rdc':
                mName = 'Forêts aléatoires [RF]'
                mName2 = 'RF'
                # model = RandomForestClassifier(class_weight='balanced', random_state=42)
                simpleimputer = SimpleImputer(add_indicator=False, copy=True, fill_value=None, missing_values=np.nan,
                                              strategy="median", verbose=0)
                standardscaler = StandardScaler(copy=True, with_mean=True, with_std=True)
                randomforestclassifier = RandomForestClassifier(bootstrap=False, class_weight=None, criterion="gini",
                                                                max_depth=None, max_features=0.21975649694764154,
                                                                max_leaf_nodes=None, min_impurity_decrease=0,
                                                                min_impurity_split=None, min_samples_leaf=2,
                                                                min_samples_split=4, min_weight_fraction_leaf=0.0,
                                                                n_estimators=300, n_jobs=1, oob_score=False,
                                                                random_state=1,
                                                                verbose=0, warm_start=False)
                model = Pipeline(memory=None,
                                 steps=[('simpleimputer', simpleimputer), ('standardscaler', standardscaler),
                                        ('randomforestclassifier', randomforestclassifier)])
                print("RandomForestClassifier\n")
            elif mods == 'dtc':
                mName = 'Arbres de décision [DT]'
                mName2 = 'DT'
                model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
                print("DecisionTreeClassifier\n")
            elif mods == 'gbc':
                mName = 'Gradient boosting [GB]'
                mName2 = 'GB'
                model = GradientBoostingClassifier()
                print("GradientBoostingClassifier\n")
            elif mods == 'etc':
                mName = 'Extremely randomized tress [ERT]'
                mName2 = 'ERT'
                model = ExtraTreesClassifier(class_weight='balanced', random_state=42)
                print("ExtraTreesClassifier\n")
            elif mods == 'abc':
                mName = 'Ada Boosting [AB]'
                mName2 = 'AB'
                model = AdaBoostClassifier()
                print("Ada Boosting\n")
            elif mods == 'bac':
                mName = 'Bootstrap Aggregating [BA]'
                mName2 = 'BA'
                model = BaggingClassifier()
                print("Bootstrap Aggregating\n")
            elif mods == 'lda':
                mName = 'Analyse discriminante linéaire [LDA]'
                mName2 = 'LDA'
                model = LinearDiscriminantAnalysis()
                print("LinearDiscriminantAnalysis\n")
            elif mods == 'qda':
               mName = 'Analyse discriminante quadratique [QDA]'
               mName2 = 'QDA'
               model = QuadraticDiscriminantAnalysis()
               print("QuadraticDiscriminantAnalysis\n")
            elif mods == 'gnb':
                mName = 'Classification naïve bayésienne [GNB]'
                mName2 = 'GNB'
                model = GaussianNB()
                print("GaussianNB\n")
            elif mods == 'rrc':
                mName = 'Régression Ridge [RR]'
                mName2 = 'RR'
                model = RidgeClassifier(class_weight='balanced')
                print("RidgeClassifier\n")
            else:
                mName = 'Régression logistique [LR]'
                mName2 = 'LR'
                model = LogisticRegression(solver='liblinear', C=10.0, class_weight='balanced')
                print("LogisticRegression\n")

            originalclass = []
            predictedclass = []
            bin_array = []

            k = model_selection.StratifiedKFold(5)
            for train_index, test_index in k.split(X, y):  # Split in X

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                if mods == ('knn' or 'dct' or 'gbc' or 'lda' or 'qda' or 'adc' or 'bac'):
                    if mods == 'knn':
                        model = KNeighborsClassifier(n_neighbors=int(len(X_train) ** (1 / 2)))
                    sm = SMOTE(sampling_strategy='auto')
                    X_train, y_train = sm.fit_resample(X_train, y_train)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                bin_array.extend(y_test)

                self.__classification_report_with_accuracy_score(y_test, y_pred, originalclass, predictedclass)

                matrix += confusion_matrix(y_test, y_pred)

            print(matrix)
            print(classification_report(originalclass, predictedclass))

            precision, recall, fscore, support = s(originalclass, predictedclass, average='macro')
            specificity_tab = []
            mNameList.append(mName)
            mNameList2.append(mName2)
            for i in range(matrix_length):
                lst2 = []
                a = self.__getTruePositive(matrix, i) / (self.__getFalsePositive(matrix, i) +
                                                         self.__getTruePositive(matrix, i))
                b = self.__getTruePositive(matrix, i) / (self.__getFalseNegative(matrix, i) +
                                                         self.__getTruePositive(matrix, i))
                c = self.__getTrueNegative(matrix, i) / (self.__getFalsePositive(matrix, i) +
                                                         self.__getTrueNegative(matrix, i))
                d = (self.__getTruePositive(matrix, i) + self.__getTrueNegative(matrix, i)) /\
                    (self.__getTruePositive(matrix, i) + self.__getTrueNegative(matrix, i) +
                     self.__getFalsePositive(matrix, i) + self.__getFalseNegative(matrix, i))
                lst2.extend((str(u[i]), self.__getSupport(matrix, i), self.__getTruePositive(matrix, i),
                             self.__getTrueNegative(matrix, i), self.__getFalsePositive(matrix, i),
                             self.__getFalseNegative(matrix, i), d,
                             a, c, b, (a+b)/2, 2*(1/((1/a)+(1/b))), (1+(2**2))*(a*b)/((2**2)*a+b)))
                lst.append(lst2[:])
                Flist.append(lst2[:])
                mNameList2.append(mName2)
                specificity_tab.append(c)
            specificity = sum(specificity_tab)/len(specificity_tab)
            accuracy = (self.__getTotalTruePositive(matrix) + self.__getTotalTrueNegative(matrix)) /\
                       (self.__getTotalTruePositive(matrix) + self.__getTotalTrueNegative(matrix) +
                        self.__getTotalFalsePositive(matrix) + self.__getTotalFalseNegative(matrix))
            tmp = ['Total', sum(sum(matrix)), self.__getTotalTruePositive(matrix),
                   self.__getTotalTrueNegative(matrix), self.__getTotalFalsePositive(matrix),
                   self.__getTotalFalseNegative(matrix),
                   accuracy, precision, specificity, recall, (precision+recall)/2, fscore,
                   (1+(2**2))*(precision*recall)/((2**2)*precision+recall)]
            lst.append(tmp[:])
            Flist.append(tmp[:])
            lst.append([''])
            print(Flist)
            print()
        for i in range(len(lst)):
            for j in range(len(lst[i])):
                if isinstance(lst[i][j], float):
                    if j != 0:
                        lst[i][j] = float("{:.2f}".format(lst[i][j] * 100))
        for i in range(len(Flist)):
            for j in range(len(Flist[i])):
                if isinstance(Flist[i][j], float):
                    if j != 0:
                        Flist[i][j] = float("{:.2f}".format(Flist[i][j] * 100))
        Ftab = self.best(self.f7(mNameList2), Flist)
        print("Meilleur")
        print()
        print(Ftab)
        print()
        print()
        return lst, mNameList, self.f7(mNameList2), Flist, Ftab


if __name__ == '__main__':
    name = 'madelon'
    var = 'Class'
    d = data.Data(name, var, [], [])
    d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova2, originLst =\
        d.ready(deleteCols=True, dropna=True, thresholdDrop=70, createDummies=True, normalize=False)
    learn = Learning(d2, ['dtc'], target, copy3)
    learn.init()
