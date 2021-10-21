import machineLearning.preprocessing.data as data
import machineLearning.tab.tab as tab
import machineLearning.utility.utility as utility

import multiprocessing
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import os
import heapq
from collections import Counter
import time
from datetime import timedelta
import psutil

import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


class PbilDiff:

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

    def create_population(self, inds, size, probas):
        pop = np.zeros((inds, size), dtype=bool)
        for i in range(inds):
            pop[i] = np.random.rand(size) <= probas
        return pop

    def create_proba(self, size):
        return np.repeat(0.5, size)

    def update_proba(self, maxi, probas, learningRate):
        for i in range(len(probas)):
            probas[i] = probas[i]*(1.0-learningRate)+maxi[i]*learningRate
        return probas

    def mutate_proba(self, probas, mutProba, mutShift):
        for i in range(len(probas)):
            if random.uniform(0, 1) < mutProba:
                probas[i] = probas[i]*(1.0-mutShift)+random.choice([0, 1])*mutShift
        return probas

    def mutate(self, xr1, xr2, xr3, F):
        mutant = xr1.astype(np.float32) + F * (xr2.astype(np.float32) - xr3.astype(np.float32))
        # print(mutant)
        mutant = np.clip(mutant, 0, 1)
        # print(mutant)
        mutant = mutant.astype(bool)
        # print(mutant, "mutant")

        return mutant

    def write_res(self, folderName, name, mode, n_pop, n_gen, F, learning_rate, mut_proba,  mut_shift,
                  y1, y2, yX, colMax, bestScorePro, bestAPro, bestPPro, bestRPro, bestFPro, bestModelPro, bestScore,
                  bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestModel, probas, debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Apprentissage incrémental à base de population différentiel" + os.linesep +\
                 "mode: " + mode + os.linesep + "name: " + name + os.linesep +\
                 "population: " + str(n_pop) + os.linesep + "générations: " + str(n_gen) + os.linesep +\
                 "F: " + str(F) + os.linesep +\
                 "taux d'apprentissage: " + str(learning_rate) + os.linesep +\
                 "probabilité de mutation: " + str(mut_proba) + os.linesep +\
                 "magnitude de mutation: " + str(mut_shift) + os.linesep +\
                 "vecteur de probabilité final: " + str(probas) + os.linesep +\
                 "moyenne: " + str(y1) + os.linesep + "meilleur: " + str(y2) + os.linesep +\
                 "classes: " + str(yX) + os.linesep + "colonnes:" + str(colMax.tolist()) + os.linesep +\
                 "scores:" + str(bestScorePro) + os.linesep + "exactitude:" + str(bestAPro) + os.linesep +\
                 "precision:" + str(bestPPro) + os.linesep + "rappel:" + str(bestRPro) + os.linesep +\
                 "fscore:" + str(bestFPro) + os.linesep + "model:" + str(bestModelPro) + os.linesep +\
                 "meilleur score: " + str(bestScore) + os.linesep +\
                 "meilleure exactitude: " + str(bestScoreA) + os.linesep +\
                 "meilleure precision: " + str(bestScoreP) + os.linesep + "meilleur rappel: " +\
                 str(bestScoreR) + os.linesep + "meilleur fscore: " + str(bestScoreF) + os.linesep +\
                 "meilleur model: " + str(bestModel) + os.linesep + "temps total: " +\
                 str(timedelta(seconds=(time.time() - debut))) + os.linesep + "mémoire: " +\
                 str(psutil.virtual_memory()) + os.linesep + "Insertions dans le tableau: " +\
                 str(self.tab_insert) + os.linesep + "Valeur présente dans le tableau: " +\
                 str(self.tab_find) + os.linesep + "temps total: " + str(yTps) + os.linesep
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path2, folderName), 'print.txt')
        f = open(a, "w")
        f.write(out)

    def natural_selection(self, part, n_pop, n_gen, F, learning_rate, mut_proba, mut_shift, data,
                          dummiesList, createDummies, normalize, metric, x, y, besties, names, iters, times, names2):

        debut = time.time()
        print_out = ""

        for mode in part:

            folderName = mode.upper()

            utility.createDirectory(path=self.path2, folderName=folderName)

            # Les axes pour le graphique
            x1 = []
            y1 = []
            y2 = []

            x2 = []
            yX = []

            yTps = []

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

            # Initialise le vecteur de probabilité
            probas = self.create_proba(size=self.copy.columns.size - 1)

            # Initialise la population
            pop = self.create_population(inds=n_pop, size=self.copy.columns.size - 1, probas=probas)

            scores, models, inds, cols, scoresA, scoresP, scoresR, scoresF, obj = \
                utility.fitness(self=self, pop=pop, mode=mode, data=data, dummiesList=dummiesList,
                                createDummies=createDummies, normalize=normalize, metric=metric)

            self.tab_data, self.tab_vals, self.tab_insert, self.tab_find =\
                obj.tab_data, obj.tab_vals, obj.tab_insert, obj.tab_find

            bestScore = np.max(scores)
            argmax = np.argmax(scores)
            bestModel = models[argmax]
            bestInd = inds[argmax]
            bestCols = cols[argmax]
            bestScoreA = scoresA[argmax]
            bestScoreP = scoresP[argmax]
            bestScoreR = scoresR[argmax]
            bestScoreF = scoresF[argmax]

            bestScorePro.append(bestScore)
            bestModelPro.append(bestModel)
            bestIndsPro.append(bestInd)
            bestColsPro.append(list(bestCols))
            bestAPro.append(bestScoreA)
            bestPPro.append(bestScoreP)
            bestRPro.append(bestScoreR)
            bestFPro.append(bestScoreF)

            # Met à jour le vecteur de probabilité
            probas = self.update_proba(bestInd, probas, learning_rate)

            # Mutation sur le vecteur de probabilité
            probas = self.mutate_proba(probas, mut_proba, mut_shift)

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

            tps_debut = timedelta(seconds=(time.time() - debut))
            yTps.append(tps_debut.total_seconds())

            print_out = \
                print_out + mode + " génération: 0" +\
                " moyenne: " + str(np.mean(heapq.nlargest(int(n_pop/2), scores))) +\
                " meilleur: " + str(bestScore) +\
                " temps exe: " + str(timedelta(seconds=(time.time() - instant))) +\
                " temps total: " + str(tps_debut) + "\n"

            print(print_out, end="")

            generation = 0
            for generation in range(n_gen):

                instant = time.time()

                pop = self.create_population(inds=n_pop, size=self.copy.columns.size - 1, probas=probas)

                # TODO Faire en sorte de ne pas recalculer la meilleure solution
                pop = np.vstack((pop, bestInd))

                # Liste des mutants
                mutants = []

                n_mut = n_pop + 1

                for i in range(n_mut):

                    # Selection des 3 individus aléatoires de la population actuelle
                    idxs = [idx for idx in range(n_mut) if idx != i]
                    selected = np.random.choice(idxs, 3, replace=False)
                    xr1, xr2, xr3 = pop[selected]

                    # mutation
                    mutant = self.mutate(xr1, xr2, xr3, F)

                    mutants.append(mutant)

                pop = np.vstack((pop, mutants))

                scores, models, inds, cols, scoresA, scoresP, scoresR, scoresF, obj = \
                    utility.fitness(self=self, pop=pop, mode=mode, data=data, dummiesList=dummiesList,
                                    createDummies=createDummies, normalize=normalize, metric=metric)

                bestScore = np.max(scores)
                argmax = np.argmax(scores)
                bestModel = models[argmax]
                bestInd = pop[argmax]
                bestCols = cols[argmax]
                bestScoreA = scoresA[argmax]
                bestScoreP = scoresP[argmax]
                bestScoreR = scoresR[argmax]
                bestScoreF = scoresF[argmax]

                bestScorePro.append(bestScore)
                bestModelPro.append(bestModel)
                bestIndsPro.append(bestInd)
                bestColsPro.append(list(bestCols))
                bestAPro.append(bestScoreA)
                bestPPro.append(bestScoreP)
                bestRPro.append(bestScoreR)
                bestFPro.append(bestScoreF)

                probas = self.update_proba(bestInd, probas, learning_rate)

                probas = self.mutate_proba(probas, mut_proba, mut_shift)

                c = Counter(scores)

                x1.append(generation + 1)

                mean_scores = np.mean(heapq.nlargest(int(n_pop / 2), scores))
                tps_instant = timedelta(seconds=(time.time() - instant))
                tps_debut = timedelta(seconds=(time.time() - debut))
                yTps.append(tps_debut.total_seconds())

                print_out = \
                    print_out + mode + " génération: " + str(generation + 1) +\
                    " moyenne: " + str(mean_scores) +\
                    " meilleur: " + str(bestScore) +\
                    " temps exe: " + str(tps_instant) +\
                    " temps total: " + str(tps_debut) + "\n"

                print(mode + " génération: " + str(generation + 1) +
                      " moyenne: " + str(mean_scores) +
                      " meilleur: " + str(bestScore) +
                      " temps exe: " + str(tps_instant) +
                      " temps total: " + str(tps_debut))

                # La moyenne sur les n_pop/2 premiers de la population
                y1.append(np.mean(heapq.nlargest(int(n_pop/2), scores)))
                y2.append(bestScore)
                fig, ax = plt.subplots()
                ax.plot(x1, y1)
                ax.plot(x1, y2)
                ax.set_title("Evolution du score par génération (" + folderName + ")"
                             + "\nApprentissage incrémental à base de population différentiel\n" + self.data_name)
                ax.set_xlabel("génération")
                ax.set_ylabel(metric)
                ax.grid()
                ax.legend(labels=["moyenne des " + str(int(n_pop/2)) + " meilleurs: " + "{:.4f}".format(mean_scores),
                                  "Le meilleur: " + "{:.4f}".format(bestScore)],
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
                              + "\nApprentissage incrémental à base de population différentiel\n" + self.data_name)
                ax2.set_xlabel("génération")
                ax2.set_ylabel("rappel")
                ax2.grid()
                ax2.legend(labels=unique, loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plotb_' + str(n_gen) + '.png')
                b = os.path.join(os.getcwd(), a)
                # if generation == n_gen-1:
                fig2.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig2)

                fig3, ax3 = plt.subplots()
                ax3.plot(x1, yTps)
                ax3.set_title("Evolution du temps d'exécution par génération (" + folderName + ")"
                              + "\nApprentissage incrémental à base de population différentiel\n" + self.data_name)
                ax3.set_xlabel("génération")
                ax3.set_ylabel("Temps en seconde")
                ax3.grid()
                ax3.legend(labels=["Temps total: " + "{:.0f}".format(tps_debut.total_seconds())],
                           loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plotTps_' + str(n_gen) + '.png')
                b = os.path.join(os.getcwd(), a)
                fig3.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig3)

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

                self.write_res(folderName=folderName, name=self.data_name, mode=mode, n_pop=n_pop, n_gen=n_gen,
                               F=F, learning_rate=learning_rate, mut_proba=mut_proba,
                               mut_shift=mut_shift, y1=y1, y2=y2, yX=yX, colMax=colMax, bestScorePro=bestScorePro,
                               bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro, bestFPro=bestFPro,
                               bestModelPro=bestModelPro, bestScore=bestScore, bestScoreA=bestScoreA,
                               bestScoreP=bestScoreP, bestScoreR=bestScoreR, bestScoreF=bestScoreF,
                               bestModel=bestModel, probas=probas, debut=debut, out=print_out, yTps=yTps)

                if (generation % 10) == 0:
                    print("Sauvegarde du tableau actuel dans les fichiers, génération:", generation)
                    tab.dump(self.tab_data, self.tab_vals, 'tab_' + self.data_name + '_' + mode)

            arg1, arg2 = utility.getList(bestModel=modelMax, bestScore=scoreMax, bestScoreA=scoreAMax,
                                         bestScoreP=scorePMax, bestScoreR=scoreRMax, bestScoreF=scoreFMax,
                                         bestCols=colMax, indMax=indMax, unique=unique, mode=mode)

            x.put(list(arg1))
            y.put(list(arg2))
            besties.put(y2)
            names.put(folderName + ": " + "{:.4f}".format(scoreMax))
            iters.put(generation)
            times.put(yTps)
            names2.put(folderName + ": " + "{:.0f}".format(tps_debut.total_seconds()))

            tab.dump(self.tab_data, self.tab_vals, 'tab_' + self.data_name + '_' + mode)

    def init(self, n_pop, n_gen, F, learning_rate, mut_proba, mut_shift, data, dummiesList, createDummies,
             normalize, metric):

        print("#############################################################")
        print("#APPRENTISSAGE INCREMENTAL A BASE DE POPULATION DIFFERENTIEL#")
        print("#############################################################")
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
            process = multiprocessing.Process(target=self.natural_selection,
                                              args=(part, n_pop, n_gen, F, learning_rate, mut_proba,
                                                    mut_shift, data, dummiesList, createDummies, normalize, metric,
                                                    x, y, besties, names, iters, times, names2))
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

        return utility.res(heuristic="Apprentissage incrémental à base de population différentiel", x=x, y=y, z=z,
                           besties=besties, names=names, iters=iters,
                           times=times, names2=names2, metric=metric, path=self.path2,
                           n_gen=n_gen, self=self)


if __name__ == '__main__':
    createDummies = False
    normalize = False

    name = 'madelon'
    var = 'Class'
    d = data.Data(name, var, [], [])

    d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova2, originLst =\
        d.ready(deleteCols=True, dropna=True, thresholdDrop=70, createDummies=True, normalize=False)

    genetic = PbilDiff(d2, d, ['lr'], target, originLst, name)
    pop = 5
    gen = 5
    F = 1
    learning_rate = 0.1
    mut_proba = 0.2
    mut_shift = 0.05
    g1, g2, g3, g4, g5 = genetic.init(n_pop=pop, n_gen=gen, F=F, learning_rate=learning_rate,
                                      mut_proba=mut_proba, mut_shift=mut_shift, data=copy2, dummiesList=d.dummiesList,
                                      createDummies=createDummies, normalize=normalize, metric="accuracy")

