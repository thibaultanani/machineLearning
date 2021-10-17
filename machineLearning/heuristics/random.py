import machineLearning.preprocessing.data as data
import machineLearning.tab.tab as tab
import machineLearning.utility.utility as utility

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import timedelta
import psutil


class Random:

    def __init__(self, data, dataObject, listModels, target, copy, data_name):
        self.data = data
        self.dataObject = dataObject
        self.listModels = listModels
        self.target = target
        self.copy = copy
        self.path2 = os.getcwd() + '/out'
        self.tab_data = []
        self.tab_vals = []
        self.tab_insert = 0
        self.tab_find = 0
        self.data_name = data_name

    def write_res(self, folderName, name, mode, n_gen, n_neighbors, proba, y1, y2, yX, colMax, bestScore,
                  bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestModel, debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Recherche aléatoire" + os.linesep +\
                 "mode: " + mode + os.linesep + "name: " + name + os.linesep + "générations: " + str(n_gen) +\
                 os.linesep + "voisins: " + str(n_neighbors) + os.linesep + "probabilité: " + str(proba) +\
                 os.linesep + "meilleur: " + str(y1) + os.linesep + "classes: " + str(yX) + os.linesep + "colonnes:" +\
                 str(colMax.tolist()) + os.linesep + "meilleur score: " + str(bestScore) + os.linesep +\
                 "meilleure exactitude: " + str(bestScoreA) + os.linesep + "meilleure precision: " + str(bestScoreP) +\
                 os.linesep + "meilleur rappel: " + str(bestScoreR) + os.linesep + "meilleur fscore: " +\
                 str(bestScoreF) + os.linesep + "meilleur model: " + str(bestModel) + os.linesep + "temps total: " +\
                 str(timedelta(seconds=(time.time() - debut))) + os.linesep + "mémoire: " +\
                 str(psutil.virtual_memory()) + os.linesep + "Insertions dans le tableau: " + str(self.tab_insert) +\
                 os.linesep + "Valeur présente dans le tableau: " + str(self.tab_find) + os.linesep +\
                 "temps total: " + str(yTps) + os.linesep
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path2, folderName), 'print.txt')
        f = open(a, "w")
        f.write(out)

    def generate_neighbors(self, n_neighbors, proba):
        neighbors = [np.random.choice(a=[False, True], size=self.copy.columns.size - 1, p=[1-proba, proba])
                     for _ in range(n_neighbors)]
        return list(neighbors)

    def optimization(self, part, n_gen, n_neighbors, proba, data, dummiesList,
                     createDummies, normalize, metric, x, y, besties, names, iters, times, names2):

        debut = time.time()
        print_out = ""

        for mode in part:

            folderName = mode.upper()

            utility.createDirectory(path=self.path2, folderName=folderName)

            cols = self.data.drop([self.target], axis=1).columns

            iteration = 0
            tps_debut = 0

            u, c = np.unique(data[self.target], return_counts=True)
            unique = list(u)

            # Initialisation du tableau
            self.tab_data, self.tab_vals = tab.init(size=len(cols), matrix_size=len(unique),
                                                    filename='tab_' + self.data_name + '_' + mode)

            x1 = []
            y1 = []
            y2 = []

            x2 = []
            yX = []

            yTps = []

            initial_solution = np.random.choice(a=[False, True], size=self.copy.columns.size - 1, p=[1-proba, proba])
            solution = initial_solution

            accuracy, recall, precision, fscore, cols, model, obj = \
                utility.fitness2(self=self, mode=mode, solution=solution, data=data, dummiesList=dummiesList,
                                 createDummies=createDummies, normalize=normalize)

            self.tab_data, self.tab_vals, self.tab_insert, self.tab_find =\
                obj.tab_data, obj.tab_vals, obj.tab_insert, obj.tab_find

            if metric == 'accuracy' or metric == 'exactitude':
                res_sol = accuracy
            elif metric == 'recall' or metric == 'rappel':
                res_sol = recall
            elif metric == 'precision' or metric == 'précision':
                res_sol = precision
            elif metric == 'fscore':
                res_sol = fscore
            else:
                res_sol = accuracy

            best_solution = solution
            best_res = res_sol
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_fscore = fscore
            best_cols = cols
            best_model = model

            while iteration < n_gen:
                instant = time.time()
                neighbors_solutions = self.generate_neighbors(n_neighbors, proba)
                for neighbor in neighbors_solutions:
                    accuracy_n, recall_n, precision_n, fscore_n, cols_n, model_n, obj = \
                        utility.fitness2(self=self, mode=mode, solution=neighbor, data=data,
                                         dummiesList=dummiesList, createDummies=createDummies, normalize=normalize)
                    self.tab_data, self.tab_vals, self.tab_insert, self.tab_find = \
                        obj.tab_data, obj.tab_vals, obj.tab_insert, obj.tab_find
                    if metric == 'accuracy' or metric == 'exactitude':
                        res_nei = accuracy_n
                    elif metric == 'recall' or metric == 'rappel':
                        res_nei = recall_n
                    elif metric == 'precision' or metric == 'précision':
                        res_nei = precision_n
                    elif metric == 'fscore':
                        res_nei = fscore_n
                    else:
                        res_nei = accuracy_n
                    if res_nei > best_res:
                        best_solution = neighbor
                        best_res = res_nei
                        best_accuracy = accuracy_n
                        best_recall = recall_n
                        best_precision = precision_n
                        best_fscore = fscore_n
                        best_cols = cols_n
                        best_model = model_n

                tps_instant = timedelta(seconds=(time.time() - instant))
                tps_debut = timedelta(seconds=(time.time() - debut))
                yTps.append(tps_debut.total_seconds())

                print_out = \
                    print_out + "mode: " + mode +\
                    " valeur: " + str(best_res) +\
                    " itération: " + str(iteration) +\
                    " temps exe: " + str(tps_instant) +\
                    " temps total: " + str(tps_debut) + "\n"

                print("mode: " + mode +
                      " valeur: " + str(best_res) +
                      " itération: " + str(iteration) +
                      " temps exe: " + str(tps_instant) +
                      " temps total: " + str(tps_debut))

                x1.append(iteration)
                y1.append(best_res)

                tmp = []
                tmp2 = []
                for i in range(len(best_model)):
                    tmp.append(utility.getTruePositive(best_model, i) /
                               (utility.getFalseNegative(best_model, i) +
                                utility.getTruePositive(best_model, i)))
                    tmp2.append(0)
                x2.append(tmp2[:])
                yX.append(tmp[:])

                fig, ax = plt.subplots()
                ax.plot(x1, y1)
                ax.set_title("Evolution du score par génération (" + folderName + ")"
                             + "\nRecherche aléatoire\n" + self.data_name)
                ax.set_xlabel("génération")
                ax.set_ylabel(metric)
                ax.grid()
                ax.legend(labels=["Le meilleur: " + "{:.4f}".format(best_res)],
                          loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plot_' + str(n_gen) + '.png')
                b = os.path.join(os.getcwd(), a)
                #if iteration == n_gen - 1:
                fig.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig)

                fig2, ax2 = plt.subplots()

                ax2.plot(x1, yX)

                ax2.set_title("Evolution du score par génération pour chacune des classes (" + folderName + ")"
                              + "\nRecherche aléatoire\n" + self.data_name)
                ax2.set_xlabel("génération")
                ax2.set_ylabel(metric)
                ax2.grid()
                ax2.legend(labels=unique, loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plotb_' + str(n_gen) + '.png')
                b = os.path.join(os.getcwd(), a)
                #if iteration == n_gen - 1:
                fig2.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig2)


                fig3, ax3 = plt.subplots()
                ax3.plot(x1, yTps)
                ax3.set_title("Evolution du temps d'exécution par génération (" + folderName + ")"
                              + "\nRecherche aléatoire\n" + self.data_name)
                ax3.set_xlabel("génération")
                ax3.set_ylabel("Temps en seconde")
                ax3.grid()
                ax3.legend(labels=["Temps total: " + "{:.0f}".format(tps_debut.total_seconds())],
                           loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plotTps_' + str(n_gen) + '.png')
                b = os.path.join(os.getcwd(), a)
                fig3.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig3)

                iteration = iteration + 1

                if (iteration % 10) == 0:
                    print("Sauvegarde du tableau actuel dans les fichiers, itération:", iteration)
                    tab.dump(self.tab_data, self.tab_vals, 'tab_' + self.data_name + '_' + mode)

            self.write_res(folderName=folderName, name=self.data_name, mode=mode, n_gen=n_gen, n_neighbors=n_neighbors,
                           proba=proba, y1=y1, y2=y2, yX=yX, colMax=best_cols, bestScore=best_res,
                           bestScoreA=best_accuracy, bestScoreP=best_precision, bestScoreR=best_recall,
                           bestScoreF=best_fscore, bestModel=best_model, debut=debut, out=print_out, yTps=yTps)

            arg1, arg2 = utility.getList(bestModel=best_model, bestScore=best_res, bestScoreA=best_accuracy,
                                         bestScoreP=best_precision, bestScoreR=best_recall, bestScoreF=best_fscore,
                                         bestCols=best_cols, indMax=best_solution, unique=unique, mode=mode)

            x.put(list(arg1))
            y.put(list(arg2))
            besties.put(y1)
            names.put(folderName + ": " + "{:.4f}".format(best_res))
            iters.put(iteration)
            times.put(yTps)
            names2.put(folderName + ": " + "{:.0f}".format(tps_debut.total_seconds()))

            tab.dump(self.tab_data, self.tab_vals, 'tab_' + self.data_name + '_' + mode)

    def init(self, n_gen, n_neighbors, proba, data, dummiesList, createDummies, normalize, metric):

        print("#####################")
        print("#RECHERCHE ALEATOIRE#")
        print("#####################")
        print()

        x = multiprocessing.Queue()
        y = multiprocessing.Queue()
        z = multiprocessing.Queue()
        besties = multiprocessing.Queue()
        names = multiprocessing.Queue()
        iters = multiprocessing.Queue()
        times = multiprocessing.Queue()
        names2 = multiprocessing.Queue()

        if isinstance(self.listModels, str):
            if self.listModels == 'all':
                self.listModels = ['x', 'rrc', 'sgd', 'knn', 'svm', 'rbf', 'dtc', 'rdc', 'etc', 'gbc', 'abc', 'bac',
                                   'lda', 'qda', 'gnb']
            else:
                self.listModels = ['x']

        n = 9
        mods = [self.listModels[i::n] for i in range(n)]

        processes = []
        for part in mods:
            process = multiprocessing.Process(target=self.optimization,
                                              args=(part, n_gen, n_neighbors, proba, data, dummiesList,
                                                    createDummies, normalize, metric, x, y, besties,
                                                    names, iters, times, names2))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        x.put(None)
        y.put(None)
        z.put(None)
        besties.put(None)
        names.put(None)
        names2.put(None)
        iters.put(None)
        times.put(None)

        x = list(iter(x.get, None))
        y = list(iter(y.get, None))
        z = list(iter(z.get, None))
        besties = list(iter(besties.get, None))
        names = list(iter(names.get, None))
        names2 = list(iter(names2.get, None))
        iters = list(iter(iters.get, None))
        times = list(iter(times.get, None))

        return utility.res(heuristic="Random Search", x=list(x), y=list(y), z=list(z),
                           besties=list(besties), names=list(names), iters=list(iters),
                           times=list(times), names2=list(names2), metric=metric, path=self.path2,
                           n_gen=n_gen - 1, self=self)


if __name__ == '__main__':
    createDummies = False
    normalize = False

    name = 'scene'
    var = 'Urban'
    d = data.Data(name, var, [], [])

    d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova2, originLst =\
        d.ready(deleteCols=True, dropna=True, thresholdDrop=70, createDummies=True, normalize=False)

    rand = Random(d2, d, ['lr'], target, originLst, name)
    gen = 5
    nei = 5
    p = 1/2
    g1, g2, g3, g4, g5 = rand.init(n_gen=gen, n_neighbors=nei, proba=p, data=copy2,
                                   dummiesList=d.dummiesList, createDummies=createDummies, normalize=normalize,
                                   metric="accuracy")
