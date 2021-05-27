import machineLearning.preprocessing.data as data

import numpy as np
import time
from datetime import timedelta
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support as s
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed

class Learning:

    """Classe définissant la partie d'aprentissage pour un modèle"""

    def __init__(self, data, listModels, target, copy, warm_start):
        self.data = data
        self.listModels = listModels
        self.target = target
        self.copy = copy
        if warm_start:
            self.data = self.data[warm_start + [target]]

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

    def train(self, X, y, train_index, test_index, model, mode, originalclass, predictedclass):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if mode == ('knn' or 'dct' or 'gbc' or 'lda' or 'qda' or 'adc' or 'bac'):
            if mode == 'knn':
                model = KNeighborsClassifier(n_neighbors=int(len(X_train) ** (1 / 2)))
            sm = SMOTE(sampling_strategy='auto')
            X_train, y_train = sm.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.__classification_report_with_accuracy_score(y_test, y_pred, originalclass, predictedclass)
        conf_mat = confusion_matrix(y_test, y_pred)


        print(accuracy_score(y_test, y_pred))
        print(originalclass)
        print(predictedclass)

        return dict(conf_mat=conf_mat, origin=originalclass, predi=predictedclass)

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

        list_accuracies = []
        list_recalls = []

        if isinstance(self.listModels, str):
            if self.listModels == 'all':
                self.listModels = ['x', 'rrc', 'knn', 'svm', 'dtc', 'rdc', 'etc', 'bac',
                                   'lda', 'gnb']
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
                mName = 'Machine à vecteurs de support [SVM]'
                mName2 = 'SVM'
                model = LinearSVC(class_weight='balanced', random_state=1)
                print("SVM\n")
            elif mods == 'rdc':
                mName = 'Forêts aléatoires [RF]'
                mName2 = 'RF'
                # model = RandomForestClassifier(class_weight='balanced', random_state=42)
                # simpleimputer = SimpleImputer(add_indicator=False, copy=True, fill_value=None, missing_values=np.nan,
                #                               strategy="median", verbose=0)
                # standardscaler = StandardScaler(copy=True, with_mean=True, with_std=True)
                # randomforestclassifier = RandomForestClassifier(bootstrap=False, class_weight=None, criterion="gini",
                #                                                 max_depth=None, max_features=0.21975649694764154,
                #                                                 max_leaf_nodes=None, min_impurity_decrease=0,
                #                                                 min_impurity_split=None, min_samples_leaf=2,
                #                                                 min_samples_split=4, min_weight_fraction_leaf=0.0,
                #                                                 n_estimators=300, n_jobs=1, oob_score=False,
                #                                                 random_state=1,
                #                                                 verbose=0, warm_start=False)
                # model = Pipeline(memory=None,
                #                  steps=[('simpleimputer', simpleimputer), ('standardscaler', standardscaler),
                #                         ('randomforestclassifier', randomforestclassifier)])
                # model = RandomForestClassifier(n_estimators=10, bootstrap=False, random_state=1)
                model = RandomForestClassifier(n_estimators=10, bootstrap=False, class_weight='balanced',
                                               random_state=1)
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

            debut = time.time()
            k = model_selection.StratifiedKFold(5)
            mean = []
            # for train_index, test_index in k.split(X, y):  # Split in X
            #
            #     X_train, X_test = X[train_index], X[test_index]
            #     y_train, y_test = y[train_index], y[test_index]
            #
            #     if mods == ('knn' or 'dct' or 'gbc' or 'lda' or 'qda' or 'adc' or 'bac'):
            #         if mods == 'knn':
            #             model = KNeighborsClassifier(n_neighbors=int(len(X_train) ** (1 / 2)))
            #         sm = SMOTE(sampling_strategy='auto')
            #         # X_train, y_train = sm.fit_resample(X_train, y_train)
            #
            #     model.fit(X_train, y_train)
            #     y_pred = model.predict(X_test)
            #     bin_array.extend(y_test)
            #
            #     self.__classification_report_with_accuracy_score(y_test, y_pred, originalclass, predictedclass)
            #
            #     matrix += confusion_matrix(y_test, y_pred)
            #
            #     print(accuracy_score(y_test, y_pred))
            #     mean.append(accuracy_score(y_test, y_pred))
            out = Parallel(n_jobs=4, verbose=100, pre_dispatch='1.5*n_jobs')(
                delayed(self.train)(train_index=train_index, test_index=test_index, X=X, y=y, model=model, mode=mods,
                                    originalclass=originalclass, predictedclass=predictedclass)
                for train_index, test_index in k.split(X, y))

            matrix = sum([d['conf_mat'] for d in out])
            origin = [d['origin'] for d in out]
            originalclass = [item for sublist in origin for item in sublist]
            predi = [d['predi'] for d in out]
            predictedclass = [item for sublist in predi for item in sublist]
            print(matrix)
            print(originalclass)
            print(predictedclass)
            print(classification_report(originalclass, predictedclass))
            # list_accuracies.append(sum(mean)/len(mean))
            # print("mean_recall:", sum(mean)/len(mean))

            print("temps exe:", timedelta(seconds=(time.time() - debut)))

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
            list_recalls.append(recall)
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

        print("#")
        print("mean_accuracies: ", list_accuracies)
        print("mean_recalls: ", list_recalls)
        print("#")

        return lst, mNameList, self.f7(mNameList2), Flist, Ftab, list_recalls


if __name__ == '__main__':
    # name = 'madelon'
    # var = 'Class'
    # d = data.Data(name, var, [], [])
    # d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova2, originLst =\
    #     d.ready(deleteCols=True, dropna=True, thresholdDrop=70, createDummies=True, normalize=True)
    # learn = Learning(d2, ['rdc'], target, copy3, ['V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V13', 'V14', 'V15', 'V16', 'V17', 'V19', 'V20', 'V21', 'V22', 'V23', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V57', 'V58', 'V59', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V79', 'V80', 'V81', 'V83', 'V84', 'V85', 'V88', 'V89', 'V90', 'V91', 'V93', 'V94', 'V95', 'V96', 'V98', 'V100', 'V101', 'V102', 'V103', 'V104', 'V106', 'V107', 'V108', 'V111', 'V112', 'V114', 'V116', 'V117', 'V119', 'V121', 'V123', 'V124', 'V126', 'V127', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V148', 'V150', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V162', 'V163', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V178', 'V179', 'V180', 'V182', 'V183', 'V184', 'V185', 'V187', 'V188', 'V189', 'V190', 'V192', 'V193', 'V195', 'V196', 'V198', 'V199', 'V200', 'V203', 'V204', 'V206', 'V207', 'V210', 'V211', 'V212', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220', 'V222', 'V223', 'V224', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V238', 'V239', 'V242', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V254', 'V255', 'V256', 'V257', 'V258', 'V260', 'V261', 'V262', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V299', 'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V308', 'V310', 'V311', 'V312', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V333', 'V334', 'V335', 'V337', 'V338', 'V339', 'V340', 'V341', 'V342', 'V343', 'V344', 'V347', 'V348', 'V349', 'V350', 'V351', 'V352', 'V353', 'V354', 'V355', 'V356', 'V357', 'V358', 'V359', 'V360', 'V361', 'V362', 'V363', 'V364', 'V366', 'V367', 'V368', 'V369', 'V371', 'V372', 'V373', 'V374', 'V375', 'V376', 'V377', 'V378', 'V379', 'V380', 'V381', 'V382', 'V383', 'V384', 'V386', 'V388', 'V389', 'V390', 'V391', 'V392', 'V393', 'V394', 'V396', 'V397', 'V398', 'V399', 'V400', 'V401', 'V402', 'V403', 'V404', 'V405', 'V406', 'V407', 'V408', 'V409', 'V410', 'V411', 'V412', 'V414', 'V415', 'V416', 'V418', 'V419', 'V420', 'V421', 'V422', 'V424', 'V426', 'V427', 'V428', 'V429', 'V430', 'V431', 'V432', 'V433', 'V434', 'V435', 'V436', 'V437', 'V438', 'V439', 'V440', 'V441', 'V442', 'V443', 'V444', 'V445', 'V447', 'V448', 'V449', 'V450', 'V451', 'V452', 'V453', 'V454', 'V455', 'V456', 'V457', 'V458', 'V459', 'V460', 'V461', 'V462', 'V464', 'V465', 'V466', 'V467', 'V468', 'V469', 'V470', 'V473', 'V474', 'V476', 'V477', 'V479', 'V481', 'V482', 'V484', 'V485', 'V486', 'V487', 'V488', 'V489', 'V490', 'V491', 'V493', 'V494', 'V495', 'V496', 'V497', 'V498', 'V499', 'V500'])
    # learn.init()
    name = 'madelon'
    var = 'Class'
    d = data.Data(name, var, [], [])
    d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova, origin =\
        d.ready(deleteCols=True, dropna=True, thresholdDrop=60, createDummies=True, normalize=False)
    print(copy3)
    print(d2)
    print(len(copy3.columns))
    learn = Learning(d2, ['bac'], target, copy3, [])
    learn.init()

