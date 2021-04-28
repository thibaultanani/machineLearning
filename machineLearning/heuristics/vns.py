import machineLearning.preprocessing.data as data
import machineLearning.tab.tab as tab
import machineLearning.utility.utility as utility

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


class VNS:

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

    def write_res(self, folderName, mode, n_gen, n_gen_vnd, kmax,  n_neighbors, y1, y2, yX, colMax, bestScore,
                  bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestModel, debut):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        f.write("mode: " + mode + os.linesep)
        f.write("générations: " + str(n_gen) + os.linesep)
        f.write("générations vnd: " + str(n_gen_vnd) + os.linesep)
        f.write("nombre de k: " + str(kmax) + os.linesep)
        f.write("voisins: " + str(n_neighbors) + os.linesep)
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

    def generate_neighbors(self, solution, n_neighbors):
        neighbors_space = []
        for i in range(5):
            lst = []
            neighbors = [list(solution.copy()) for _ in range(n_neighbors)]
            for ind in neighbors:
                mutate_index = random.sample(range(0, len(solution)), i + 1)
                for x in mutate_index:
                    ind[x] = not ind[x]
                lst.append(ind)
            neighbors_space.append(lst)
        return neighbors_space

    def vnd(self, n_gen, kmax, n_neighbors, solution, best_res, best_solution,  best_accuracy, best_precision,
            best_recall, best_fscore, best_cols, best_model, data, mode, dummiesList, createDummies, normalize, metric):
        iteration = 0
        while iteration < n_gen:
            k = 0
            neighbor_space = self.generate_neighbors(solution, n_neighbors)
            while k < kmax:
                for n in neighbor_space[k]:
                    accuracy_n, recall_n, precision_n, fscore_n, cols_n, model_n, obj = \
                        utility.fitness2(self=self, mode=mode, solution=n, data=data, dummiesList=dummiesList,
                                         createDummies=createDummies, normalize=normalize)
                    self.tab_data, self.tab_vals, self.tab_insert, self.tab_find = \
                        obj.tab_data, obj.tab_vals, obj.tab_insert, obj.tab_find
                    if metric == 'accuracy' or 'exactitude':
                        res_nei = accuracy_n
                    elif metric == 'recall' or 'rappel':
                        res_nei = recall_n
                    elif metric == 'precision' or 'précision':
                        res_nei = precision_n
                    elif metric == 'fscore':
                        res_nei = fscore_n
                    else:
                        res_nei = accuracy_n
                    if res_nei > best_res:
                        best_solution = n
                        best_res = res_nei
                        best_accuracy = accuracy_n
                        best_precision = precision_n
                        best_recall = recall_n
                        best_fscore = fscore_n
                        best_cols = cols_n
                        best_model = model_n
                k = k + 1
            iteration = iteration + 1
        return best_res, best_solution, best_accuracy, best_precision, best_recall, best_fscore, best_cols, best_model

    def optimization(self, part, n_gen, n_gen_vnd, kmax, n_neighbors, data,
                     dummiesList, createDummies, normalize, metric, x, y, besties, names, iters):

        debut = time.time()

        for mode in part:

            folderName = mode.upper()

            utility.createDirectory(path=self.path2, folderName=folderName)

            cols = self.data.drop([self.target], axis=1).columns

            iteration = 0

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

            initial_solution = np.random.choice(a=[False, True], size=self.copy.columns.size - 1)
            solution = initial_solution

            accuracy, recall, precision, fscore, cols, model, obj = \
                utility.fitness2(self=self, mode=mode, solution=solution, data=data, dummiesList=dummiesList,
                                 createDummies=createDummies, normalize=normalize)

            self.tab_data, self.tab_vals, self.tab_insert, self.tab_find =\
                obj.tab_data, obj.tab_vals, obj.tab_insert, obj.tab_find

            if metric == 'accuracy' or 'exactitude':
                res_sol = accuracy
            elif metric == 'recall' or 'rappel':
                res_sol = recall
            elif metric == 'precision' or 'précision':
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
                k = 0
                neighbor_space = self.generate_neighbors(solution, n_neighbors)
                while k < kmax:
                    for solution_prime in neighbor_space[k]:
                        res_nei, res_sol, accuracy_n, precision_n, recall_n, fscore_n, cols_n, model_n = self.vnd(
                            n_gen_vnd, kmax, n_neighbors, solution_prime, best_res, best_solution, best_accuracy,
                            best_precision,
                            best_recall, best_fscore, best_cols, best_model, data, mode, dummiesList, createDummies,
                            normalize,
                            metric)

                        if metric == 'accuracy' or 'exactitude':
                            res_nei = accuracy_n
                        elif metric == 'recall' or 'rappel':
                            res_nei = recall_n
                        elif metric == 'precision' or 'précision':
                            res_nei = precision_n
                        elif metric == 'fscore':
                            res_nei = fscore_n
                        else:
                            res_nei = accuracy_n

                        if res_nei > best_res:
                            best_solution = res_sol
                            best_res = res_nei
                            best_accuracy = accuracy_n
                            best_precision = precision_n
                            best_recall = recall_n
                            best_fscore = fscore_n
                            best_cols = cols_n
                            best_model = model_n

                    k = k + 1

                print("mode: ", mode, " valeur: ", best_res, " iteration: ", iteration,
                      " temps exe: ", str(timedelta(seconds=(time.time() - instant))),
                      " temps total: ", str(timedelta(seconds=(time.time() - debut))))

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
                             + "\nRecherche à voisinage variable")
                ax.set_xlabel("génération")
                ax.set_ylabel(metric)
                ax.grid()
                ax.legend(labels=["Le meilleur"],
                          loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plot_' + str(n_gen) + '.png')
                b = os.path.join(os.getcwd(), a)
                #if iteration == n_gen - 1:
                fig.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig)

                fig2, ax2 = plt.subplots()

                ax2.plot(x1, yX)

                ax2.set_title("Evolution du score par génération pour chacune des classes (" + folderName + ")"
                              + "\nRecherche locale itérée")
                ax2.set_xlabel("génération")
                ax2.set_ylabel(metric)
                ax2.grid()
                ax2.legend(labels=unique, loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plotb_' + str(n_gen) + '.png')
                b = os.path.join(os.getcwd(), a)
                #if iteration == n_gen - 1:
                fig2.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig2)

                iteration = iteration + 1

                self.write_res(folderName=folderName, mode=mode, n_gen=n_gen, n_gen_vnd=n_gen_vnd,
                               kmax=kmax, n_neighbors=n_neighbors, y1=y1, y2=y2, yX=yX, colMax=best_cols,
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
            besties.put(y1)
            names.put(folderName + ": " + "{:.3f}".format(best_res))
            iters.put(iteration)

            tab.dump(self.tab_data, self.tab_vals, 'tab_' + self.data_name + '_' + mode)

    def init(self, n_gen, n_gen_vnd, kmax, n_neighbors, data, dummiesList, createDummies, normalize, metric):

        print("#################################")
        print("#RECHERCHE A VOISINNAGE VARIABLE#")
        print("#################################")
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
                                      args=(part, n_gen, n_gen_vnd, kmax, n_neighbors, data, dummiesList,
                                            createDummies, normalize, metric, x, y, besties, names, iters))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return utility.res(heuristic="Recherche à voisinage variable", x=list(x.queue), y=list(y.queue),
                           z=list(z.queue), besties=list(besties.queue), names=list(names.queue),
                           iters=list(iters.queue), metric=metric, path=self.path2, n_gen=n_gen-1, self=self)


if __name__ == '__main__':
    createDummies = False
    normalize = False

    name = 'madelon'
    var = 'Class'
    d = data.Data(name, var, [], [])

    d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova2, originLst =\
        d.ready(deleteCols=True, dropna=True, thresholdDrop=70, createDummies=True, normalize=False)

    vns = VNS(d2, d, ['lr'], target, originLst, name)
    gen = 5
    gen_vnd = 2
    nei = 2
    kmax = 2
    g1, g2, g3, g4, g5 = vns.init(n_gen=gen, n_gen_vnd=gen_vnd, kmax=kmax, n_neighbors=nei, data=copy2,
                                  dummiesList=d.dummiesList, createDummies=createDummies, normalize=normalize,
                                  metric="accuracy")
