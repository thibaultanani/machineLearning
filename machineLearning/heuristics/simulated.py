import machineLearning.preprocessing.data as data
import machineLearning.tab.tab as tab
import machineLearning.utility.utility as utility

import math
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
from datetime import timedelta
import psutil

import warnings
warnings.filterwarnings('ignore')


class Simulated:

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

    def write_res(self, folderName, mode, temperature, alpha, final_temperature, y1, y2, yX, colMax, bestScore,
                  bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestModel, debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "mode: " + mode + os.linesep + "température initiale: " + str(temperature) + os.linesep + "alpha: " +\
                 str(alpha) + os.linesep + "température finale: " + str(final_temperature) + os.linesep +\
                 "meilleur: " + str(y1) + os.linesep + "classes: " + str(yX) + os.linesep + "colonnes:" +\
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

    def optimization(self, part, temperature, alpha, final_temperature, data,
                     dummiesList, createDummies, normalize, metric, x, y, besties, names, iters, times, names2):

        debut = time.time()
        print_out = ""

        for mode in part:

            folderName = mode.upper()

            utility.createDirectory(path=self.path2, folderName=folderName)

            iteration = 0
            tps_debut = 0

            cols = self.data.drop([self.target], axis=1).columns

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

            solution = np.random.choice(a=[False, True], size=self.copy.columns.size - 1)

            best_solution = None
            best_res = 0
            best_accuracy = None
            best_precision = None
            best_recall = None
            best_fscore = None
            best_cols = None
            best_model = None

            begin_temperature = temperature

            while temperature > final_temperature:
                instant = time.time()

                mutate_index = random.sample(range(0, len(solution)), 1)
                neighbor = solution.copy()
                for m in mutate_index:
                    neighbor[m] = not neighbor[m]

                accuracy, recall, precision, fscore, cols, model, obj = \
                    utility.fitness2(self, mode, solution, data, dummiesList, createDummies, normalize)

                self.tab_data, self.tab_vals, self.tab_insert, self.tab_find = \
                    obj.tab_data, obj.tab_vals, obj.tab_insert, obj.tab_find

                accuracy_n, recall_n, precision_n, fscore_n, cols_n, model_n, obj = \
                    utility.fitness2(self, mode, neighbor, data, dummiesList, createDummies, normalize)

                self.tab_data, self.tab_vals, self.tab_insert, self.tab_find = \
                    obj.tab_data, obj.tab_vals, obj.tab_insert, obj.tab_find

                if metric == 'accuracy' or metric == 'exactitude':
                    res_sol = accuracy
                    res_nei = accuracy_n
                elif metric == 'recall' or metric == 'rappel':
                    res_sol = recall
                    res_nei = recall_n
                elif metric == 'precision' or metric == 'précision':
                    res_sol = precision
                    res_nei = precision_n
                elif metric == 'fscore':
                    res_sol = fscore
                    res_nei = fscore_n
                else:
                    res_sol = accuracy
                    res_nei = accuracy_n

                if res_sol > best_res:
                    best_solution = solution
                    best_res = res_sol
                    best_accuracy = accuracy
                    best_precision = precision
                    best_recall = recall
                    best_fscore = fscore
                    best_cols = cols
                    best_model = model

                cost = res_nei - res_sol
                if cost >= 0:
                    solution = neighbor
                    res_sol = res_nei
                else:
                    r = random.uniform(0, 1)
                    if r < math.exp(- cost / temperature):
                        solution = neighbor
                        res_sol = res_nei

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
                y2.append(res_sol)

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
                ax.plot(x1, y2)
                ax.set_title("Evolution du score par génération (" + folderName + ")"
                             + "\nRecuit simulé")
                ax.set_xlabel("génération")
                ax.set_ylabel(metric)
                ax.grid()
                ax.legend(labels=["Le meilleur: " + "{:.3f}".format(best_res),
                                  "Valeur actuelle: " + "{:.3f}".format(res_sol)],
                          loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plot_' + str(begin_temperature) + '.png')
                b = os.path.join(os.getcwd(), a)
                # if iteration == begin_temperature - 1:
                fig.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig)

                fig2, ax2 = plt.subplots()

                ax2.plot(x1, yX)

                ax2.set_title("Evolution du score par génération pour chacune des classes (" + folderName + ")"
                              + "\nRecuit simulé")
                ax2.set_xlabel("génération")
                ax2.set_ylabel(metric)
                ax2.grid()
                ax2.legend(labels=unique, loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plotb_' + str(begin_temperature) + '.png')
                b = os.path.join(os.getcwd(), a)
                # if iteration == begin_temperature - 1:
                fig2.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig2)

                fig3, ax3 = plt.subplots()
                ax3.plot(x1, yTps)
                ax3.set_title("Evolution du temps d'exécution par génération (" + folderName + ")"
                              + "\nRecuit suimulé")
                ax3.set_xlabel("génération")
                ax3.set_ylabel("Temps en seconde")
                ax3.grid()
                ax3.legend(labels=["Temps total: " + "{:.0f}".format(tps_debut.total_seconds())],
                           loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plotTps_' + str(begin_temperature) + '.png')
                b = os.path.join(os.getcwd(), a)
                fig3.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig3)

                temperature = temperature - alpha
                iteration = iteration + 1

                if (iteration % 10) == 0:
                    print("Sauvegarde du tableau actuel dans les fichiers, itération:", iteration)
                    tab.dump(self.tab_data, self.tab_vals, 'tab_' + self.data_name + '_' + mode)

            temperature = begin_temperature

            self.write_res(folderName=folderName, mode=mode, temperature=begin_temperature, alpha=alpha,
                           final_temperature=final_temperature, y1=y1, y2=y2, yX=yX, colMax=best_cols,
                           bestScore=best_res, bestScoreA=best_accuracy, bestScoreP=best_precision,
                           bestScoreR=best_recall, bestScoreF=best_fscore, bestModel=best_model, debut=debut,
                           out=print_out, yTps=yTps)

            arg1, arg2 = utility.getList(bestModel=best_model, bestScore=best_res, bestScoreA=best_accuracy,
                                         bestScoreP=best_precision, bestScoreR=best_recall, bestScoreF=best_fscore,
                                         bestCols=best_cols, indMax=best_solution, unique=unique, mode=mode)

            x.put(list(arg1))
            y.put(list(arg2))
            besties.put(y1)
            names.put(folderName + ": " + "{:.3f}".format(best_res))
            iters.put(iteration)
            times.put(yTps)
            names2.put(folderName + ": " + "{:.0f}".format(tps_debut.total_seconds()))

            tab.dump(self.tab_data, self.tab_vals, 'tab_' + self.data_name + '_' + mode)

    def init(self, temperature, alpha, final_temperature, data, dummiesList, createDummies, normalize, metric):

        print("###############")
        print("#RECUIT SIMULE#")
        print("###############")
        print()

        x = queue.Queue()
        y = queue.Queue()
        z = queue.Queue()
        besties = queue.Queue()
        names = queue.Queue()
        iters = queue.Queue()
        times = queue.Queue()
        names2 = queue.Queue()

        if isinstance(self.listModels, str):
            if self.listModels == 'all':
                self.listModels = ['x', 'rrc', 'sgd', 'knn', 'svm', 'rbf', 'dtc', 'rdc', 'etc', 'gbc', 'abc', 'bac',
                                   'lda', 'qda', 'gnb']
            else:
                self.listModels = ['x']

        n = 4
        mods = [self.listModels[i::n] for i in range(n)]

        threads = []
        for part in mods:
            thread = threading.Thread(target=self.optimization,
                                      args=(part, temperature, alpha, final_temperature, data, dummiesList,
                                            createDummies, normalize, metric, x, y, besties,
                                            names, iters, times, names2))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return utility.res(heuristic="Recuit simulé", x=list(x.queue), y=list(y.queue), z=list(z.queue),
                           besties=list(besties.queue), names=list(names.queue), iters=list(iters.queue),
                           times=list(times.queue), names2=list(names2.queue), metric=metric, path=self.path2,
                           n_gen=temperature-1, self=self)


if __name__ == '__main__':
    createDummies = False
    normalize = False

    name = 'madelon'
    var = 'Class'
    d = data.Data(name, var, [], [])

    d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova2, originLst =\
        d.ready(deleteCols=True, dropna=True, thresholdDrop=70, createDummies=True, normalize=False)

    simulated = Simulated(d2, d, ['lr'], target, originLst, name)
    temperature = 10
    alpha = 1
    final = 0
    mut = copy.columns.size - 1
    g1, g2, g3, g4, g5 = simulated.init(temperature=10, alpha=1, final_temperature=0, data=copy2,
                                        dummiesList=d.dummiesList, createDummies=createDummies, normalize=normalize,
                                        metric="accuracy")
