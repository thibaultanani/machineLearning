import machineLearning.preprocessing.data as data
import machineLearning.tab.tab as tab
import machineLearning.utility.utility as utility

import threading
import queue
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import os
import heapq
from operator import itemgetter
from collections import Counter
import time
from datetime import timedelta
import psutil

import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


class ACO:

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
        self.pheromones_proba = [0.5]*(copy.columns.size-1)
        self.pheromones_tau = [0.5]*(copy.columns.size-1)

    def create_population(self, ants, size):
        pop = np.zeros((ants, size), dtype=bool)
        for i in range(ants):
            for j in range(size):
                r = random.uniform(0, 1)
                if r < self.pheromones_tau[j]:
                    pop[i, j] = False
                else:
                    pop[i, j] = True
        return pop

    def update_pheromones(self, pop, scores, p, phi, theta, alpha, best_score_index):
        new_pheromones_tau = []
        sum_pheromones_tau = [0]*(self.copy.columns.size-1)
        for ant, score in zip(pop, scores):
            sum_pheromones_tau = [x + y for x, y in zip(sum_pheromones_tau,
                                                        self.evaporate_pheromones(ant, score, phi, theta))]
        for i in range(len(self.pheromones_tau)):
            new_pheromones_tau.append((1 - p)*self.pheromones_tau[i] + sum_pheromones_tau[i] +
                                      sum_pheromones_tau[best_score_index])
        self.getProba(pheromone_tau=new_pheromones_tau, alpha=alpha)

    def evaporate_pheromones(self, ant, score, phi, theta):
        delta = []
        n = len(ant)
        solution_size = sum(ant)
        for x in ant:
            if x:
                delta.append(phi * score + ((theta*(n-solution_size))/n))
            else:
                delta.append(0)
        return delta

    def getProba(self, pheromone_tau, alpha):
        s = max([x**alpha for x in pheromone_tau])
        for i in range(len(pheromone_tau)):
            self.pheromones_tau[i] = pheromone_tau[i]**alpha/s

    def write_res(self, folderName, mode, n_pop, n_gen, p, phi, alpha, y1, y2, yX, colMax, bestScorePro, bestAPro,
                  bestPPro, bestRPro, bestFPro, bestModelPro, bestScore, bestScoreA, bestScoreP, bestScoreR, bestScoreF,
                  bestModel, debut):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "mode: " + mode + os.linesep + "population: " + str(n_pop) + os.linesep +\
                 "générations: " + str(n_gen) + os.linesep + "p: " + str(p) + os.linesep + "phi: " + str(phi) +\
                 os.linesep + "alpha: " + str(alpha) + os.linesep + "moyenne: " + str(y1) + os.linesep + "meilleur: " +\
                 str(y2) + os.linesep + "classes: " + str(yX) + os.linesep + "colonnes:" + str(colMax.tolist()) +\
                 os.linesep + "scores:" + str(bestScorePro) + os.linesep + "exactitude:" + str(bestAPro) + os.linesep +\
                 "precision:" + str(bestPPro) + os.linesep + "rappel:" + str(bestRPro) + os.linesep +\
                 "fscore:" + str(bestFPro) + os.linesep + "model:" + str(bestModelPro) + os.linesep +\
                 "meilleur score: " + str(bestScore) + os.linesep + "meilleure exactitude: " + str(bestScoreA) +\
                 os.linesep + "meilleure precision: " + str(bestScoreP) + os.linesep + "meilleur rappel: " +\
                 str(bestScoreR) + os.linesep + "meilleur fscore: " + str(bestScoreF) + os.linesep +\
                 "meilleur model: " + str(bestModel) + os.linesep + "temps total: " +\
                 str(timedelta(seconds=(time.time() - debut))) + os.linesep + "mémoire: " +\
                 str(psutil.virtual_memory()) + os.linesep + "Insertions dans le tableau: " +\
                 str(self.tab_insert) + os.linesep + "Valeur présente dans le tableau: " +\
                 str(self.tab_find) + os.linesep
        f.write(string)
        f.close()

    def optimization(self, part, n_pop, n_gen, p, phi, alpha, data, dummiesList,
                          createDummies, normalize, metric, x, y, besties, names, iters):

        debut = time.time()

        theta = 1 - phi

        for mode in part:

            folderName = mode.upper()

            utility.createDirectory(path=self.path2, folderName=folderName)

            # Les axes pour le graphique
            x1 = []
            y1 = []
            y2 = []

            x2 = []
            yX = []

            scoreMax = 0
            modelMax = 0
            indMax = 0
            colMax = 0
            scoreAMax = 0
            scorePMax = 0
            scoreRMax = 0
            scoreFMax = 0

            cols = self.data.drop([self.target], axis=1).columns

            u, c = np.unique(data[self.target], return_counts=True)
            unique = list(u)

            # Initialisation du tableau
            self.tab_data, self.tab_vals = tab.init(size=len(cols), matrix_size=len(unique),
                                                    filename='tab_' + self.data_name + '_' + mode)

            # Progression des meilleurs éléments
            bestScorePro = []
            bestModelPro = []
            bestIndsPro = []
            bestColsPro = []
            bestAPro = []
            bestPPro = []
            bestRPro = []
            bestFPro = []

            # Mesurer le temps d'execution
            instant = time.time()

            # Initialise la population de forumis
            pop = self.create_population(ants=n_pop, size=self.copy.columns.size-1)

            scores, models, inds, cols, scoresA, scoresP, scoresR, scoresF, obj = \
                utility.fitness(self=self, pop=pop, mode=mode, data=data, dummiesList=dummiesList,
                                createDummies=createDummies, normalize=normalize, metric=metric)

            self.tab_data, self.tab_vals, self.tab_insert, self.tab_find =\
                obj.tab_data, obj.tab_vals, obj.tab_insert, obj.tab_find

            bestScore = np.max(scores)
            bestModel = models[np.argmax(scores)]
            bestInd = inds[np.argmax(scores)]
            bestCols = cols[np.argmax(scores)]
            bestScoreA = scoresA[np.argmax(scores)]
            bestScoreP = scoresP[np.argmax(scores)]
            bestScoreR = scoresR[np.argmax(scores)]
            bestScoreF = scoresF[np.argmax(scores)]

            self.update_pheromones(pop=pop, scores=scores, p=p, phi=phi, theta=theta, alpha=alpha,
                                   best_score_index=np.argmax(scores))

            bestScorePro.append(bestScore)
            bestModelPro.append(bestModel)
            bestIndsPro.append(bestInd)
            bestColsPro.append(list(bestCols))
            bestAPro.append(bestScoreA)
            bestPPro.append(bestScoreP)
            bestRPro.append(bestScoreR)
            bestFPro.append(bestScoreF)

            x1.append(0)
            y1.append(np.mean(heapq.nlargest(int(n_pop/2), scores)))
            y2.append(bestScore)

            tmp = []
            tmp2 = []
            for i in range(len(bestModel)):
                tmp.append(utility.getTruePositive(bestModel, i) / (utility.getFalseNegative(bestModel, i) +
                                                                    utility.getTruePositive(bestModel, i)))
                tmp2.append(0)
            x2.append(tmp2[:])
            yX.append(tmp[:])

            print("mode: ", mode, " valeur: ", str(bestScore), " génération: 0",
                  " temps exe: ", str(timedelta(seconds=(time.time() - instant))),
                  " temps total: ", str(timedelta(seconds=(time.time() - debut))))

            generation = 0
            for generation in range(n_gen):

                instant = time.time()

                # Nouvelle génération de fourmis
                pop = self.create_population(ants=n_pop, size=self.copy.columns.size - 1)

                scores, models, inds, cols, scoresA, scoresP, scoresR, scoresF, obj = \
                    utility.fitness(self=self, pop=pop, mode=mode, data=data, dummiesList=dummiesList,
                                    createDummies=createDummies, normalize=normalize, metric=metric)

                self.tab_data, self.tab_vals, self.tab_insert, self.tab_find = \
                    obj.tab_data, obj.tab_vals, obj.tab_insert, obj.tab_find

                bestScore = np.max(scores)
                bestModel = models[np.argmax(scores)]
                bestInd = pop[np.argmax(scores)]
                bestCols = cols[np.argmax(scores)]
                bestScoreA = scoresA[np.argmax(scores)]
                bestScoreP = scoresP[np.argmax(scores)]
                bestScoreR = scoresR[np.argmax(scores)]
                bestScoreF = scoresF[np.argmax(scores)]
                bestScorePro.append(bestScore)
                bestModelPro.append(bestModel)
                bestIndsPro.append(bestInd)
                bestColsPro.append(list(bestCols))
                bestAPro.append(bestScoreA)
                bestPPro.append(bestScoreP)
                bestRPro.append(bestScoreR)
                bestFPro.append(bestScoreF)

                self.update_pheromones(pop=pop, scores=scores, p=p, phi=phi, theta=theta, alpha=alpha,
                                       best_score_index=np.argmax(scores))

                c = Counter(scores)

                x1.append(generation + 1)

                print("mode: ", mode, " valeur: ", str(max(bestScorePro)), " génération: ", generation + 1,
                      " temps exe: ", str(timedelta(seconds=(time.time() - instant))),
                      " temps total: ", str(timedelta(seconds=(time.time() - debut))))

                # La moyenne sur les n_pop/2 premiers de la population
                y1.append(bestScore)
                y2.append(max(bestScorePro))
                fig, ax = plt.subplots()
                ax.plot(x1, y1)
                ax.plot(x1, y2)
                ax.set_title("Evolution du score par génération (" + folderName + ")"
                             + "\nOptimisation par colonie de fourmis")
                ax.set_xlabel("génération")
                ax.set_ylabel(metric)
                ax.grid()
                ax.legend(labels=["Valeur actuelle", "Le meilleur"],
                          loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plot_' + str(n_gen) + '.png')
                b = os.path.join(os.getcwd(), a)
                # if generation == n_gen - 1:
                fig.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig)

                fig2, ax2 = plt.subplots()

                tmp = []
                tmp2 = []
                for i in range(len(bestModel)):
                    tmp.append(utility.getTruePositive(bestModel, i) / (utility.getFalseNegative(bestModel, i) +
                                                                        utility.getTruePositive(bestModel, i)))
                    tmp2.append(generation + 1)
                yX.append(tmp[:])
                x2.append(tmp2[:])

                ax2.plot(x1, yX)

                ax2.set_title("Evolution du rappel par génération pour chacune des classes (" + folderName + ")"
                              + "\nOptimisation par colonie de fourmis")
                ax2.set_xlabel("génération")
                ax2.set_ylabel("rappel")
                ax2.grid()
                ax2.legend(labels=unique, loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plotb_' + str(n_gen) + '.png')
                b = os.path.join(os.getcwd(), a)
                # if generation == n_gen-1:
                fig2.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig2)

                generation = generation + 1

                if bestScore > scoreMax:
                    scoreMax = bestScore
                    modelMax = bestModel
                    indMax = bestInd
                    colMax = bestCols
                    scoreAMax = bestScoreA
                    scorePMax = bestScoreP
                    scoreRMax = bestScoreR
                    scoreFMax = bestScoreF

                self.write_res(folderName=folderName, mode=mode, n_pop=n_pop, n_gen=n_gen, p=p, phi=phi, alpha=alpha,
                               y1=y1, y2=y2, yX=yX, colMax=colMax, bestScorePro=bestScorePro, bestAPro=bestAPro,
                               bestPPro=bestPPro, bestRPro=bestRPro, bestFPro=bestFPro, bestModelPro=bestModelPro,
                               bestScore=bestScore, bestScoreA=bestScoreA, bestScoreP=bestScoreP,
                               bestScoreR=bestScoreR, bestScoreF=bestScoreF, bestModel=bestModel, debut=debut)

                if (generation % 5) == 0:
                    print("Sauvegarde du tableau actuel dans les fichiers, génération:", generation)
                    tab.dump(self.tab_data, self.tab_vals, 'tab_' + self.data_name + '_' + mode)

            arg1, arg2 = utility.getList(bestModel=modelMax, bestScore=scoreMax, bestScoreA=scoreAMax,
                                         bestScoreP=scorePMax, bestScoreR=scoreRMax, bestScoreF=scoreFMax,
                                         bestCols=colMax, indMax=indMax, unique=unique, mode=mode)

            x.put(list(arg1))
            y.put(list(arg2))
            besties.put(y2)
            names.put(folderName + ": " + "{:.3f}".format(scoreMax))
            iters.put(generation)

            tab.dump(self.tab_data, self.tab_vals, 'tab_' + self.data_name + '_' + mode)

    def init(self, n_pop, n_gen, p, phi, alpha, data, dummiesList, createDummies, normalize, metric):

        print("####################################")
        print("#OPTIMSATION PAR COLONIE DE FOURMIS#")
        print("####################################")
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
                                      args=(part, n_pop, n_gen, p, phi, alpha, data, dummiesList,
                                            createDummies, normalize, metric, x, y, besties, names, iters))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return utility.res(heuristic="Optimisation par colonie de forumis", x=list(x.queue), y=list(y.queue),
                           z=list(z.queue), besties=list(besties.queue), names=list(names.queue),
                           iters=list(iters.queue), metric=metric, path=self.path2, n_gen=n_gen, self=self)


if __name__ == '__main__':
    createDummies = False
    normalize = False

    name = 'madelon'
    var = 'Class'
    d = data.Data(name, var, [], [])

    d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova2, originLst =\
        d.ready(deleteCols=True, dropna=True, thresholdDrop=70, createDummies=True, normalize=False)

    aco = ACO(d2, d, ['lr'], target, originLst, name)
    pop = 40
    gen = 10
    p = 0.5
    phi = 0.8
    alpha = 2
    g1, g2, g3, g4, g5 = aco.init(n_pop=pop, n_gen=gen, p=p, phi=phi, alpha=alpha, data=copy2,
                                  dummiesList=d.dummiesList, createDummies=createDummies, normalize=normalize,
                                  metric="accuracy")
