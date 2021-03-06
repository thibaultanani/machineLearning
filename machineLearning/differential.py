import machineLearning.data as data
import machineLearning.tab as tab
import machineLearning.utility as utility

import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
import os
import heapq
from collections import Counter
import time
from datetime import timedelta
import psutil


class Differential:

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

    def crossover(self, ind, mutant, cross_proba):
        cross_points = np.random.rand(self.copy.columns.size-1) <= cross_proba
        # print(cross_points, "cross_points")

        trial = np.where(cross_points, mutant, ind)
        # print(trial, "trial")
        # print(ind, "original")

        idxs = [idx for idx in range(len(ind))]
        selected = np.random.choice(idxs, 1, replace=False)

        trial[selected] = not trial[selected]

        return trial

    def mutate(self, pop, xr1, xr2, xr3, F):
        mutant = xr1.astype(np.float32) + F * (xr2.astype(np.float32) - xr3.astype(np.float32))
        # print(mutant)
        mutant = np.clip(mutant, 0, 1)
        # print(mutant)
        mutant = mutant.astype(bool)
        # print(mutant, "mutant")

        return mutant

    def write_res(self, folderName, mode, n_pop, n_gen, cross_proba, F, y1, y2, yX, colMax, bestScorePro, bestAPro, bestPPro,
                  bestRPro, bestFPro, bestModelPro, bestScore, bestScoreA, bestScoreP, bestScoreR, bestScoreF,
                  bestModel, debut):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        f.write("mode: " + mode + os.linesep)
        f.write("population: " + str(n_pop) + os.linesep)
        f.write("g??n??rations: " + str(n_gen) + os.linesep)
        f.write("probabilit?? de crisement: " + str(cross_proba) + os.linesep)
        f.write("F: " + str(F) + os.linesep)
        f.write("moyenne: " + str(y1) + os.linesep)
        f.write("meilleur: " + str(y2) + os.linesep)
        f.write("classes: " + str(yX) + os.linesep)
        f.write("colonnes:" + str(colMax.tolist()) + os.linesep)
        f.write("scores:" + str(bestScorePro) + os.linesep)
        f.write("exactitude:" + str(bestAPro) + os.linesep)
        f.write("precision:" + str(bestPPro) + os.linesep)
        f.write("rappel:" + str(bestRPro) + os.linesep)
        f.write("fscore:" + str(bestFPro) + os.linesep)
        f.write("model:" + str(bestModelPro) + os.linesep)
        f.write("meilleur score: " + str(bestScore) + os.linesep)
        f.write("meilleure exactitude: " + str(bestScoreA) + os.linesep)
        f.write("meilleure precision: " + str(bestScoreP) + os.linesep)
        f.write("meilleur rappel: " + str(bestScoreR) + os.linesep)
        f.write("meilleur fscore: " + str(bestScoreF) + os.linesep)
        f.write("meilleur model: " + str(bestModel) + os.linesep)
        f.write("temps total: " + str(timedelta(seconds=(time.time() - debut))) + os.linesep)
        f.write("m??moire: " + str(psutil.virtual_memory()) + os.linesep)
        f.write("Insertions dans le tableau: " + str(self.tab_insert) + os.linesep)
        f.write("Valeur pr??sente dans le tableau: " + str(self.tab_find) + os.linesep)
        f.close()

    def selection(self, pop, mutants, n_pop):
        pop_list = []
        for ind, score, model, col, scoreA, scoreP, scoreR, scoreF in pop:
            pop_list.append(list([list(ind), score, model, col, scoreA, scoreP, scoreR, scoreF]))
        mut_list = []
        for ind, score, model, col, scoreA, scoreP, scoreR, scoreF in mutants:
            mut_list.append(list([list(ind), score, model, col, scoreA, scoreP, scoreR, scoreF]))
        newpop = []
        scores = []
        models = []
        cols = []
        scoresA = []
        scoresP = []
        scoresR = []
        scoresF = []
        for i in range(n_pop):
            if pop_list[i][1] > mut_list[i][1]:
                newpop.append((pop_list[i][0]))
                scores.append((pop_list[i][1]))
                models.append((pop_list[i][2]))
                cols.append((pop_list[i][3]))
                scoresA.append((pop_list[i][4]))
                scoresP.append((pop_list[i][5]))
                scoresR.append((pop_list[i][6]))
                scoresF.append((pop_list[i][7]))
            else:
                newpop.append((mut_list[i][0]))
                scores.append((mut_list[i][1]))
                models.append((mut_list[i][2]))
                cols.append((mut_list[i][3]))
                scoresA.append((mut_list[i][4]))
                scoresP.append((mut_list[i][5]))
                scoresR.append((mut_list[i][6]))
                scoresF.append((mut_list[i][7]))
        return np.array(newpop), scores, models, cols, scoresA, scoresP, scoresR, scoresF

    def natural_selection(self, part, n_pop, n_gen, cross_proba, F, data, dummiesList,
                          createDummies, normalize, metric, x, y, besties, names, iters):

        debut = time.time()

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

            # Progression des meilleurs ??l??ments
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

            # Initialise notre population
            pop = utility.create_population(inds=n_pop, size=self.copy.columns.size - 1)

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

            bestScorePro.append(bestScore)
            bestModelPro.append(bestModel)
            bestIndsPro.append(bestInd)
            bestColsPro.append(bestCols)
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

            print(mode + " g??n??ration: 0" +
                  " moyenne: " + str(np.mean(heapq.nlargest(int(n_pop/2), scores))) + " meilleur: " + str(bestScore) +
                  " temps exe: " + str(timedelta(seconds=(time.time() - instant))) +
                  " temps total: " + str(timedelta(seconds=(time.time() - debut))))

            generation = 0
            for generation in range(n_gen):

                instant = time.time()

                # Liste des mutants
                mutants = []

                for i in range(n_pop):

                    # Selection des 3 individus al??atoires de la population actuelle
                    idxs = [idx for idx in range(n_pop) if idx != i]
                    selected = np.random.choice(idxs, 3, replace=False)
                    xr1, xr2, xr3 = pop[selected]

                    # mutation
                    mutant = self.mutate(pop, xr1, xr2, xr3, F)

                    # croisement
                    trial = self.crossover(pop[i], mutant, cross_proba)

                    mutants.append(trial)

                # Calcul du score pour l'ensemble des mutants
                scores_m, models_m, inds_m, cols_m, scoresA_m, scoresP_m, scoresR_m, scoresF_m, obj = \
                    utility.fitness(self=self, pop=mutants, mode=mode, data=data, dummiesList=dummiesList,
                                    createDummies=createDummies, normalize=normalize, metric=metric)

                self.tab_data, self.tab_vals, self.tab_insert, self.tab_find = \
                    obj.tab_data, obj.tab_vals, obj.tab_insert, obj.tab_find

                # selection des meilleurs individus
                pop_score = zip(pop, scores, models, cols, scoresA, scoresP, scoresR, scoresF)
                mut_score = zip(mutants, scores_m, models_m, cols_m, scoresA_m, scoresP_m, scoresR_m, scoresF_m)

                pop, scores, models, cols, scoresA, scoresP, scoresR, scoresF = \
                    self.selection(pop_score, mut_score, n_pop)

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

                c = Counter(scores)

                print(mode + " g??n??ration: " + str(generation + 1) +
                      " moyenne: " + str(np.mean(heapq.nlargest(int(n_pop / 2), scores))) +
                      " meilleur: " + str(bestScore) +
                      " temps exe: " + str(timedelta(seconds=(time.time() - instant))) +
                      " temps total: " + str(timedelta(seconds=(time.time() - debut))))

                x1.append(generation + 1)
                # La moyenne sur les n_pop/2 premiers de la population
                y1.append(np.mean(heapq.nlargest(int(n_pop/2), scores)))
                y2.append(bestScore)
                fig, ax = plt.subplots()
                ax.plot(x1, y1)
                ax.plot(x1, y2)
                ax.set_title("Evolution du score par g??n??ration (" + folderName + ")")
                ax.set_xlabel("g??n??ration")
                ax.set_ylabel(metric)
                ax.grid()
                ax.legend(labels=["moyenne des " + str(int(n_pop/2)) + " meilleurs", "Le meilleur"],
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

                ax2.set_title("Evolution du score par g??n??ration pour chacune des classes (" + folderName + ")")
                ax2.set_xlabel("g??n??ration")
                ax2.set_ylabel(metric)
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

                self.write_res(folderName=folderName, mode=mode, n_pop=n_pop, n_gen=n_gen, cross_proba=cross_proba,
                               F=F, y1=y1, y2=y2, yX=yX, colMax=colMax, bestScorePro=bestScorePro, bestAPro=bestAPro,
                               bestPPro=bestPPro, bestRPro=bestRPro, bestFPro=bestFPro, bestModelPro=bestModelPro,
                               bestScore=bestScore, bestScoreA=bestScoreA, bestScoreP=bestScoreP,
                               bestScoreR=bestScoreR, bestScoreF=bestScoreF, bestModel=bestModel, debut=debut)

                if (generation % 5) == 0:
                    print("Sauvegarde du tableau actuel dans les fichiers, g??n??ration:", generation)
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

    def init(self, n_pop, n_gen, cross_proba, F, data, dummiesList, createDummies, normalize, metric):

        print("#######################################")
        print("#ALGORITHME A EVOLUTION DIFFERENTIELLE#")
        print("#######################################")
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
            thread = threading.Thread(target=self.natural_selection,
                                      args=(part, n_pop, n_gen, cross_proba, F, data, dummiesList,
                                            createDummies, normalize, metric, x, y, besties, names, iters))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return utility.res(heuristic="Evolution diff??rentielle", x=list(x.queue), y=list(y.queue), z=list(z.queue),
                           besties=list(besties.queue), names=list(names.queue), iters=list(iters.queue),
                           metric=metric, path=self.path2, n_gen=n_gen, self=self)


if __name__ == '__main__':
    createDummies = False
    normalize = False

    name = 'madelon'
    var = 'Class'
    d = data.Data(name, var, [], [])

    d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova2, originLst = \
        d.ready(deleteCols=True, dropna=True, thresholdDrop=70, createDummies=True, normalize=False)

    genetic = Differential(d2, d, ['lr'], target, originLst, name)
    pop = 5
    gen = 5
    cross_proba = 0.5
    F = 1
    g1, g2, g3, g4, g5 = genetic.init(n_pop=pop, n_gen=gen, cross_proba=cross_proba, F=F,
                                      data=copy2, dummiesList=d.dummiesList, createDummies=createDummies,
                                      normalize=normalize, metric="accuracy")

