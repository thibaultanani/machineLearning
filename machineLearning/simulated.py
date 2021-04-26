import machineLearning.data as data
import machineLearning.tab as tab
import machineLearning.utility as utility

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
        self.path2 = os.path.dirname(os.getcwd()) + '/out'
        self.tab_data = []
        self.tab_vals = []
        self.tab_insert = 0
        self.tab_find = 0
        self.data_name = data_name

    def write_res(self, folderName, mode, temperature, alpha, final_temperature, y1, y2, yX, colMax, bestScore,
                  bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestModel, debut):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        f.write("mode: " + mode + os.linesep)
        f.write("température initiale: " + str(temperature) + os.linesep)
        f.write("alpha: " + str(alpha) + os.linesep)
        f.write("température finale: " + str(final_temperature) + os.linesep)
        f.write("meilleur: " + str(y1) + os.linesep)
        f.write("classes: " + str(yX) + os.linesep)
        f.write("colonnes:" + str(colMax.tolist()) + os.linesep)
        f.write("meilleur score: " + str(bestScore) + os.linesep)
        f.write("meilleure exactitude: " + str(bestScoreA) + os.linesep)
        f.write("meilleure precision: " + str(bestScoreP) + os.linesep)
        f.write("meilleur rappel: " + str(bestScoreR) + os.linesep)
        f.write("meilleur fscore: " + str(bestScoreF) + os.linesep)
        f.write("meilleur model: " + str(bestModel) + os.linesep)
        f.write("temps total: " + str(timedelta(seconds=(time.time() - debut))) + os.linesep)
        f.write("mémoire: " + str(psutil.virtual_memory()) + os.linesep)
        f.write("Insertions dans le tableau: " + str(self.tab_insert) + os.linesep)
        f.write("Valeur présente dans le tableau: " + str(self.tab_find) + os.linesep)
        f.close()

    def optimization(self, part, temperature, alpha, final_temperature, data,
                     dummiesList, createDummies, normalize, metric, x, y, besties, names, iters):

        debut = time.time()

        for mode in part:

            folderName = mode.upper()

            utility.createDirectory(path=self.path2, folderName=folderName)

            iteration = 0

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

                if metric == 'accuracy' or 'exactitude':
                    res_sol = accuracy
                    res_nei = accuracy_n
                elif metric == 'recall' or 'rappel':
                    res_sol = recall
                    res_nei = recall_n
                elif metric == 'precision' or 'précision':
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

                print("mode: ", mode, " valeur: ", best_res, " iteration: ", iteration,
                      " temps exe: ", str(timedelta(seconds=(time.time() - instant))),
                      " temps total: ", str(timedelta(seconds=(time.time() - debut))))

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
                ax.legend(labels=["Le meilleur", "Valeur actuelle"],
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

                temperature = temperature - alpha
                iteration = iteration + 1

            self.write_res(folderName=folderName, mode=mode, temperature=begin_temperature, alpha=alpha,
                           final_temperature=final_temperature, y1=y1, y2=y2, yX=yX, colMax=best_cols,
                           bestScore=best_res, bestScoreA=best_accuracy, bestScoreP=best_precision,
                           bestScoreR=best_recall, bestScoreF=best_fscore, bestModel=best_model, debut=debut)

            if (iteration % 5) == 0:
                print("Sauvegarde du tableau actuel dans les fichiers, itération:", iteration)
                tab.dump(self.tab_data, self.tab_vals, 'tab_' + self.data_name + '_' + mode)

            arg1, arg2 = utility.getList(bestModel=best_model, bestScore=best_res, bestScoreA=best_accuracy,
                                         bestScoreP=best_precision, bestScoreR=best_recall, bestScoreF=best_fscore,
                                         bestCols=best_cols, indMax=best_solution, unique=unique, mode=mode)

            x.put(list(arg1))
            y.put(list(arg2))
            besties.put(y2)
            names.put(folderName + ": " + "{:.3f}".format(best_res))
            iters.put(iteration)

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
                                            createDummies, normalize, metric, x, y, besties, names, iters))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return utility.res(heuristic="Recuit simulé", x=list(x.queue), y=list(y.queue), z=list(z.queue),
                           besties=list(besties.queue), names=list(names.queue), iters=list(iters.queue),
                           metric=metric, path=self.path2, n_gen=temperature-1, self=self)


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
