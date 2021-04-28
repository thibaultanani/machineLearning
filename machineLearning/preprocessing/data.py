import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
import statsmodels.api as sm


class Data:

    """Classe définissant un jeu de données et qui permet de le traiter"""

    def __init__(self, name, target, dropColsList, dropClassList):
        self.path = os.path.join(os.getcwd() + '/in', name)
        self.target = target
        self.dropColsList = dropColsList
        self.dropClassList = dropClassList
        self.dummiesList = None
        self.path2 = os.getcwd() + '/out'

    def createDummiesLst(self, data):
        col = [c for c in data.select_dtypes(exclude=['bool']).columns if c != self.target]
        dummies_lst = []
        for c in col:
            if c.endswith('quali'):
                dummies_lst.append(c)
        return dummies_lst

    # Lecture d'un fichier xlsx
    @staticmethod
    def __read(filename):
        try:
            data = pd.read_excel(filename + '.xlsx', index_col=None)
        except:
            data = pd.read_csv(filename + '.csv', index_col=None, sep=',')
        return data

    # Retorune un dataframe normalisé
    @staticmethod
    def __normalize(data):
        data = (data - data.min()) / (data.max() - data.min())
        return data.fillna(0)

    # Retorune le nombre de NaN pour chaque colonne
    @staticmethod
    def __sumNanList(data):
        list1 = []
        for i in range(data.columns.size):
            count = 0
            count = data[data.columns[i]].isna().sum()
            list1.append(count)
        return list1

    # Retourne la liste des colonnes
    @staticmethod
    def __columnsNameList(data):
        nameList = []
        for col in data.columns:
            nameList.append(col)
        return nameList

    # Retourne le pourcentage de NaN pour chaque colonne
    @staticmethod
    def __ratioList(data, sumNanList):
        list1 = []
        rowsCount = len(data)
        for i in range(data.columns.size):
            list1.append("{0:.2f}".format(100 * sumNanList[i] / rowsCount))
        return list1

    # Supprime les colonnes qui ont un nombre de NaN superieur a un seuil
    def deleteCol(self, data, threshold):
        list1 = self.__sumNanList(data)
        list2 = self.__columnsNameList(data)
        list3 = self.__ratioList(data, list1)
        list4 = []
        for i in range(data.columns.size):
            if float(list3[i]) > threshold:
                list4.append(list2[i])
        tmp = data.drop(list4, axis=1)
        return tmp

    # Sélectionne les colonnes du dataframe aléatoirement
    @staticmethod
    def __selectionAlea(data, target):
        tmp = data
        for col in data.columns:
            if random.randint(0, 1) == 0 and col != target:
                tmp = tmp.drop(col, axis=1)
        return tmp

    # Créer k-1 variables indicatrices pour les variables qualitatives si k > 2. k = valeurs différentes possibles
    def createDummies(self, data):
        dummiesList = self.dummiesList
        list1 = []
        for col in dummiesList:
            if col in data.columns:
                list1.append(col)
        data = pd.get_dummies(data, columns=list1)
        return data, list1

    def createDirectoryOriginal(self):
        final = self.path2
        if os.path.exists(final):
            shutil.rmtree(final)
        os.makedirs(final)

    def createDirectory(self, folderName):
        final = os.path.join(self.path2, folderName)
        if os.path.exists(final):
            shutil.rmtree(final)
        os.makedirs(final)

    def display(self, contingencyList, dependencyList, data):
        print("########################")
        print("#TABLEAU DE CONTINGENCE#")
        print("########################")
        for contingency in contingencyList:
            print()
            print(contingency)
        print()
        print("##############")
        print("#CHI2 / ANOVA#")
        print("##############")
        ctmp = self.__columnsNameList(data)
        i = 0
        for dependency in dependencyList:
            print()
            if not isinstance(dependency, pd.DataFrame):
                tmp = [[ctmp[i], dependency[0], dependency[2], dependency[1], dependency[3]]]
                print(pd.DataFrame(tmp, columns=['Variable', 'Chi2', 'Dgr. liberté', 'p-valeur', 'Résultat']))
            else:
                print(dependency)
            i = i + 1
        print()
        print()

    def corr(self, data):
        print()
        print("########################")
        print("#MATRICE DE CORRELATION#")
        print("########################")
        print()
        folderName = 'Corr'
        self.createDirectory(folderName)
        data = data.drop(self.target, axis=1)
        for c in data.columns:
            if c == self.target:
                data = data.drop(c, axis=1)
        corr = data.corr(method='pearson')
        print(len(corr))
        if len(corr) < 20:
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(corr, annot=True, xticklabels=True, yticklabels=True, cmap="Greens", vmin=-1,
                        vmax=1, annot_kws={"size": 7}, fmt=".2f")
            ax.tick_params(labelsize=8)
            file = os.path.join(os.path.join(self.path2, folderName), folderName + '0.png')
            plt.savefig(file, bbox_inches='tight')
            plt.close()
            plt.show()
        else:
            k = 0
            for i in range(0, len(corr), 20):
                if i + 20 > len(corr):
                    limit = len(corr)
                else:
                    limit = i + 20
                for j in range(0, len(corr), 20):
                    k = k + 1
                    if j + 20 <= len(corr) and limit <= len(corr):
                        square = corr.iloc[i:limit, j:j+20]
                    else:
                        square = corr.iloc[i:limit, j:len(corr)]
                    print(square.to_string())
                    print()
                    fig, ax = plt.subplots(figsize=(15, 10))
                    sns.heatmap(square, annot=True, xticklabels=True, yticklabels=True, cmap="Greens", vmin=-1,
                                vmax=1, annot_kws={"size": 7}, fmt=".2f")
                    ax.tick_params(labelsize=7)
                    file = os.path.join(os.path.join(self.path2, folderName), folderName + str(k) + '.png')
                    plt.savefig(file, bbox_inches='tight')
                    plt.close()
                    plt.show()

    def chi2(self, data):
        folderName = 'Chi2'
        self.createDirectory(folderName)
        if len(data[self.target].unique()) < 3:
            cmap1 = ['dodgerblue', 'salmon', 'springgreen']
        else:
            cmap1 = ['dodgerblue', 'springgreen', 'yellow', 'orange', 'salmon']
        contingencyList = []
        dependencyList = []
        for c in data.columns:
            if (c in self.dummiesList or data[c].dtypes == np.bool or data[c].dtypes == np.object) and c != self.target:
                table = pd.crosstab(data[c], data[self.target])
                table_para = pd.crosstab(data[c], data[self.target], margins=True, margins_name="Total")

                contingencyList.append(table_para)
                chi2, p, dof, ex = chi2_contingency(table, correction=False)
                if p <= 0.05:
                    test = 'Rejet'
                else:
                    test = 'Acceptation'
                tmp = [float("{:.3f}".format(chi2)), float("{:.2f}".format(p)), dof, test]
                dependencyList.extend([tmp])

                plt.figure(figsize=(20, 3))
                ax = table.plot.bar(stacked=False, color=cmap1, width=0.7)
                plt.grid()
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=self.target)
                plt.title(c + ' et ' + self.target)
                file = os.path.join(os.path.join(self.path2, folderName), c + '.png')
                plt.savefig(file, bbox_inches='tight')
                plt.close('all')
            else:
                if c != self.target:
                    ndata = pd.cut(data[c], 3, right=False)
                    ndata = ndata.astype(str).str.replace(')', '')
                    ndata = ndata.astype(str).str.replace('[', '')
                    ndata = ndata.astype(str).str.replace(',', ' -')
                    table_para = pd.crosstab(ndata, data[self.target], margins=True, margins_name='Total')
                    contingencyList.append(table_para)
                    s = str(c) + ' ~ ' + str(self.target)
                    model = ols(s, data=data).fit()
                    aov_table = sm.stats.anova_lm(model, typ=2)
                    aov_table = aov_table.rename(index={self.target: c + '~' + self.target})
                    aov_table = aov_table.round(3)
                    dependencyList.append(aov_table)
                    plt.figure(figsize=(10, 5))
                    sns.boxplot(x=c, y=self.target, data=data, color='white', orient='horizontal')
                    sns.stripplot(x=c, y=self.target, data=data, orient='horizontal', palette=cmap1)
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=np.unique(data[self.target]))
                    plt.title(c + ' et ' + self.target)
                    plt.grid()
                    plt.tight_layout()
                    file = os.path.join(os.path.join(self.path2, folderName), c + '.png')
                    plt.savefig(file, bbox_inches='tight')
                    plt.close('all')
        self.display(contingencyList, dependencyList, data)
        return contingencyList, dependencyList

    # Transforme l'ensemble des données pour être prêt pour la sélection naturelle
    def ready(self, deleteCols, dropna, thresholdDrop, createDummies, normalize):
        data = self.__read(filename=self.path)
        data = data[~data[self.target].isin(self.dropClassList)]
        copy = data.copy()
        data = data.drop(self.dropColsList, axis=1)
        self.createDirectoryOriginal()
        if deleteCols:
            data = self.deleteCol(data=data, threshold=thresholdDrop)
        copy2 = data.copy()
        if dropna:
            data = data.dropna()
        for col in data.columns:
            result = all(elem in [True, False] for elem in data[col].unique())
            if result:
                data[col] = data[col].astype(bool)
        copy3 = data.copy()
        self.dummiesList = self.createDummiesLst(data=data)
        # chi2, anova = self.chi2(copy3)
        chi2, anova = None, None
        # self.corr(copy3)
        if createDummies:
            data, dummiesLst = self.createDummies(data=data)
            origin, x = self.createDummies(data=copy2)
        else:
            dummiesLst = None
            origin = data.copy()
        copy4 = data.copy()
        if normalize:
            data = self.__normalize(data=data)
        return data, self.target, copy, copy2, copy3, copy4, dummiesLst, thresholdDrop, chi2, anova, origin


if __name__ == '__main__':
    d = Data('mdcolab', 'covid19_test_results',
             ['test_name', 'swab_type'], [])
    d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova, origin =\
        d.ready(deleteCols=True, dropna=True, thresholdDrop=70, createDummies=True, normalize=False)
    print(d2)
    print(d.dummiesList)
    print(dummiesLst)
