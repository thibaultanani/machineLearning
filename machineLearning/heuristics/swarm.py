import machineLearning.preprocessing.data as data
import machineLearning.tab.tab as tab
import machineLearning.utility.utility as utility

import threading
import queue
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from imblearn.over_sampling import SMOTE
import os
from sklearn.metrics import precision_recall_fscore_support as s
import time
from datetime import timedelta
import psutil

import warnings
warnings.filterwarnings('ignore')


class Particle:

    def __init__(self, dimension):
        self.position = np.random.choice(a=[False, True], size=dimension)
        self.pbest_position = self.position
        self.pbest_value = 0
        self.pbest_accuracy = 0
        self.pbest_precision = 0
        self.pbest_recall = 0
        self.pbest_fscore = 0
        self.pbest_column = []
        self.pbest_matrix = None
        self.velocity = np.random.choice(a=[False, True], size=dimension)
        self.w = np.random.choice(a=[False, True], size=dimension)
        self.c1 = np.random.choice(a=[False, True], size=dimension)
        self.c2 = np.random.choice(a=[False, True], size=dimension)

    def __str__(self):
        return "Je suis a la position " + str(self.position) + " et le pbest est " + str(self.pbest_position)

    def move(self):
        self.position = self.position ^ self.velocity


class Swarm:

    def __init__(self, n_particles, dimension, data, target, mode, dummiesList, createDummies, normalize, w, c1, c2):
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = 0
        self.gbest_accuracy = 0
        self.gbest_precision = 0
        self.gbest_recall = 0
        self.gbest_fscore = 0
        self.gbest_position = np.random.choice(a=[False, True], size=dimension)
        self.gbest_column = []
        self.gbest_matrix = None
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.data = data
        self.target = target
        self.mode = mode
        self.dummiesList = dummiesList
        self.createDummies = createDummies
        self.normalize = normalize
        self.tab_data = []
        self.tab_vals = []
        self.tab_insert = 0
        self.tab_find = 0

    def print_particles(self):
        for particle in self.particles:
            print(particle.position)

    def fitness(self, particle, metric):

        matrix_length = len(np.unique(self.data[self.target]))

        if self.mode == 'sgd':
            model = SGDClassifier(class_weight='balanced', loss='modified_huber', random_state=1)
        elif self.mode == 'svr':
            model = SVC(kernel='linear', class_weight='balanced', probability=True)
        elif self.mode == 'rdf':
            model = SVC(kernel='rbf', class_weight='balanced', probability=True)
        elif self.mode == 'pol':
            model = SVC(kernel='poly', class_weight='balanced', probability=True)
        elif self.mode == 'rdc':
            model = RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=1)
        elif self.mode == 'dtc':
            model = DecisionTreeClassifier(class_weight='balanced', random_state=1)
        elif self.mode == 'gdc':
            model = GradientBoostingClassifier(random_state=1)
        elif self.mode == 'etc':
            model = ExtraTreesClassifier(class_weight='balanced', random_state=1)
        elif self.mode == 'adc':
            model = AdaBoostClassifier(random_state=1)
        elif self.mode == 'bac':
            model = BaggingClassifier(random_state=1)
        elif self.mode == 'lda':
            model = LinearDiscriminantAnalysis()
        elif self.mode == 'qda':
            model = QuadraticDiscriminantAnalysis()
        elif self.mode == 'gnb':
            model = GaussianNB()
        elif self.mode == 'rrc':
            model = RidgeClassifier(class_weight='balanced')
        else:
            model = LogisticRegression(solver='liblinear', C=10.0, class_weight='balanced')
        k = model_selection.StratifiedKFold(5)
        try:
            tab_data, tab_val = tab.get([int(x) for x in particle.posiion], self.tab_data, self.tab_vals)
            tab_val = np.array(tab_val)
            accuracy = (utility.getTotalTruePositive(tab_val) + utility.getTotalTrueNegative(tab_val)) / \
                       (utility.getTotalTruePositive(tab_val) + utility.getTotalTrueNegative(tab_val) +
                        utility.getTotalFalsePositive(tab_val) + utility.getTotalFalseNegative(tab_val))
            precision_tab = []
            recall_tab = []
            for i in range(len(tab_val)):
                a = utility.getTruePositive(tab_val, i) / (utility.getFalsePositive(tab_val, i) +
                                                           utility.getTruePositive(tab_val, i))
                b = utility.getTruePositive(tab_val, i) / (utility.getFalseNegative(tab_val, i) +
                                                           utility.getTruePositive(tab_val, i))
                precision_tab.append(a)
                recall_tab.append(b)
            precision = sum(precision_tab) / len(precision_tab)
            recall = sum(recall_tab) / len(recall_tab)
            fscore = 2 * (1 / ((1 / precision) + (1 / recall)))
            matrix = tab_val
            tmp = self.data.drop([self.target], axis=1)
            tmp = tmp.iloc[:, particle.position]
            cols = tmp.columns
            self.tab_find = self.tab_find + 1
        except AttributeError:
            matrix = np.zeros((matrix_length, matrix_length), dtype=int)
            X, y, cols = utility.ready(self, particle.position, self.data, self.dummiesList, self.createDummies,
                                       self.normalize)
            originalclass = []
            predictedclass = []
            for train_index, test_index in k.split(X, y):  # Split in X
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if self.mode == ('knn' or 'dct' or 'gbc' or 'lda' or 'qda' or 'adc' or 'bac'):
                    if self.mode == 'knn':
                        model = KNeighborsClassifier(n_neighbors=int(len(X_train) ** (1 / 2)))
                    sm = SMOTE(sampling_strategy='auto')
                    X_train, y_train = sm.fit_resample(X_train, y_train)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                originalclass.extend(y_test)
                predictedclass.extend(y_pred)

                matrix += confusion_matrix(y_test, y_pred)

            accuracy = (utility.getTotalTruePositive(matrix) + utility.getTotalTrueNegative(matrix)) / \
                       (utility.getTotalTruePositive(matrix) + utility.getTotalTrueNegative(matrix) +
                        utility.getTotalFalsePositive(matrix) + utility.getTotalFalseNegative(matrix))

            precision, recall, fscore, support = s(originalclass, predictedclass, average='macro')
            self.tab_data, self.tab_vals = tab.add([int(x) for x in particle.position], matrix.tolist(), self.tab_data,
                                                   self.tab_vals)
            self.tab_insert = self.tab_insert + 1

        if metric == 'accuracy' or 'exactitude':
            score = accuracy
        elif metric == 'recall' or 'rappel':
            score = recall
        elif metric == 'precision' or 'précision':
            score = precision
        elif metric == 'fscore':
            score = fscore
        else:
            score = accuracy

        return score, accuracy, recall, precision, fscore, cols, matrix

    def set_pbest(self, metric):
        for particle in self.particles:
            fitness_candidate, accuracy, recall, precision, fscore, cols, model = self.fitness(particle, metric)
            if particle.pbest_value < fitness_candidate:
                particle.pbest_value = fitness_candidate
                particle.pbest_accuracy = accuracy
                particle.pbest_precision = precision
                particle.pbest_recall = recall
                particle.pbest_fscore = fscore
                particle.pbest_matrix = model
                particle.pbest_position = particle.position
                particle.pbest_column = cols

    def set_gbest(self, metric):
        # TODO Réfléchir sur l'utilité de calculer 2 fois l'ensemble des particules
        for particle in self.particles:
            best_fitness_candidate, best_accuracy, best_recall, best_precision, \
            best_fscore, best_cols, best_model = self.fitness(particle, metric)
            if self.gbest_value < best_fitness_candidate:
                self.gbest_value = best_fitness_candidate
                self.gbest_accuracy = best_accuracy
                self.gbest_precision = best_precision
                self.gbest_recall = best_recall
                self.gbest_fscore = best_fscore
                self.gbest_matrix = best_model
                self.gbest_position = particle.position
                self.gbest_column = best_cols

    def move_particles(self):
        for particle in self.particles:

            # w * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)] + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

            # Calcul de w * v_{ij}(t)
            # w est la probabilité qu'une variable soit prise
            particle.w = np.random.rand(len(particle.w)) <= self.w
            a = particle.velocity & particle.w

            # Calcul de c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
            # c1 est la probabilité qu'une variable soit prise
            particle.c1 = np.random.rand(len(particle.c1)) <= self.c1
            b = particle.c1 & (particle.pbest_position ^ particle.position)

            # Calcul de c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]
            # c2 est la probabilité qu'une variable soit prise
            particle.c2 = np.random.rand(len(particle.c2)) <= self.c2
            c = particle.c2 & (self.gbest_position ^ particle.position)

            particle.velocity = a | b | c
            particle.move()


class PSO:

    def __init__(self, data, dataObject, listModels, target, copy, data_name):
        self.data = data
        self.dataObject = dataObject
        self.listModels = listModels
        self.target = target
        self.copy = copy
        self.path2 = os.getcwd() + '/out'
        self.data_name = data_name

    def write_res(self, folderName, mode, n_pop, n_gen, w, c1, c2, y1, yX, colMax,
                  bestScore, bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestModel, debut, insert, find):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        f.write("mode: " + mode + os.linesep)
        f.write("population: " + str(n_pop) + os.linesep)
        f.write("générations: " + str(n_gen) + os.linesep)
        f.write("w: " + str(w) + os.linesep)
        f.write("c1: " + str(c1) + os.linesep)
        f.write("c2: " + str(c2) + os.linesep)
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
        f.write("Insertions dans le tableau: " + str(insert) + os.linesep)
        f.write("Valeur présente dans le tableau: " + str(find) + os.linesep)
        f.close()

    def optimization(self, part, n_pop, n_gen, w, c1, c2, data, dummiesList, createDummies, normalize, metric, x, y,
                     besties, names, iters):

        debut = time.time()

        for mode in part:

            folderName = mode.upper()

            utility.createDirectory(path=self.path2, folderName=folderName)

            search_space = Swarm(n_pop, data.columns.size - 1, data, self.target, mode, dummiesList, createDummies,
                                 normalize, w, c1, c2)
            particles_vector = [Particle(data.columns.size - 1) for _ in range(search_space.n_particles)]
            search_space.particles = particles_vector

            cols = self.data.drop([self.target], axis=1).columns

            u, c = np.unique(data[self.target], return_counts=True)
            unique = list(u)

            # Initialisation du tableau
            search_space.tab_data, search_space.tab_vals = tab.init(size=len(cols), matrix_size=len(unique),
                                                                    filename='tab_' + self.data_name + '_' + mode)

            x1 = []
            y1 = []

            x2 = []
            yX = []

            iteration = 0
            while iteration < n_gen:
                instant = time.time()

                search_space.set_pbest(metric)
                search_space.set_gbest(metric)

                search_space.move_particles()

                print("mode: ", mode, " valeur: ", search_space.gbest_value, " itération: ", iteration,
                      " temps exe: ", str(timedelta(seconds=(time.time() - instant))),
                      " temps total: ", str(timedelta(seconds=(time.time() - debut))))

                x1.append(iteration)
                y1.append(search_space.gbest_value)

                tmp = []
                tmp2 = []
                for i in range(len(search_space.gbest_matrix)):
                    tmp.append(utility.getTruePositive(search_space.gbest_matrix, i) /
                               (utility.getFalseNegative(search_space.gbest_matrix, i) +
                                utility.getTruePositive(search_space.gbest_matrix, i)))
                    tmp2.append(0)
                x2.append(tmp2[:])
                yX.append(tmp[:])

                fig, ax = plt.subplots()
                ax.plot(x1, y1)
                # ax.plot(x1, y2)
                ax.set_title("Evolution du score par génération (" + folderName + ")"
                             + "\nOptimisation par essaim de particule")
                ax.set_xlabel("génération")
                ax.set_ylabel(metric)
                ax.grid()
                ax.legend(labels=["Le meilleur"],
                          loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plot_' + str(1) + '.png')
                b = os.path.join(os.getcwd(), a)
                # if iteration == n_gen - 1:
                fig.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig)

                fig2, ax2 = plt.subplots()

                ax2.plot(x1, yX)

                ax2.set_title("Evolution du score par génération pour chacune des classes (" + folderName + ")"
                              + "\nOptimisation par essaim de particule")
                ax2.set_xlabel("génération")
                ax2.set_ylabel(metric)
                ax2.grid()
                ax2.legend(labels=unique, loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
                a = os.path.join(os.path.join(self.path2, folderName), 'plotb_' + str(1) + '.png')
                b = os.path.join(os.getcwd(), a)
                # if iteration == n_gen-1:
                fig2.savefig(os.path.abspath(b), bbox_inches="tight")
                plt.close(fig2)

                iteration = iteration + 1

                self.write_res(folderName=folderName, mode=mode, n_pop=n_pop, n_gen=n_gen, w=w, c1=c1, c2=c2, y1=y1,
                               yX=yX, colMax=search_space.gbest_column, bestScore=search_space.gbest_value,
                               bestScoreA=search_space.gbest_accuracy, bestScoreP=search_space.gbest_precision,
                               bestScoreR=search_space.gbest_recall, bestScoreF=search_space.gbest_fscore,
                               bestModel=search_space.gbest_matrix, debut=debut,
                               insert=search_space.tab_insert, find=search_space.tab_find)

                if (iteration % 5) == 0:
                    print("Sauvegarde du tableau actuel dans les fichiers, itération:", iteration)
                    tab.dump(search_space.tab_data, search_space.tab_vals, 'tab_' + self.data_name + '_' + mode)

            arg1, arg2 = utility.getList(bestModel=search_space.gbest_matrix, bestScore=search_space.gbest_value,
                                         bestScoreA=search_space.gbest_accuracy,
                                         bestScoreP=search_space.gbest_precision, bestScoreR=search_space.gbest_recall,
                                         bestScoreF=search_space.gbest_fscore, bestCols=search_space.gbest_column,
                                         indMax=search_space.gbest_position, unique=unique, mode=mode)

            x.put(list(arg1))
            y.put(list(arg2))
            besties.put(y1)
            names.put(folderName + ": " + "{:.3f}".format(search_space.gbest_value))
            iters.put(iteration)

            tab.dump(search_space.tab_data, search_space.tab_vals, 'tab_' + self.data_name + '_' + mode)

    def init(self, n_pop, n_gen, w, c1, c2, data, dummiesList, createDummies, normalize, metric):

        print("#####################################")
        print("#OPTIMSATION PAR ESSAIM DE PARTICULE#")
        print("#####################################")
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
                                      args=(part, n_pop, n_gen, w, c1, c2, data, dummiesList,
                                            createDummies, normalize, metric, x, y, besties, names, iters))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return utility.res(heuristic="Optimisation par essaim de particules", x=list(x.queue), y=list(y.queue),
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

    pso = PSO(d2, d, ['lr'], target, originLst, name)
    pop = 5
    gen = 5
    w = 0.5
    c1 = 0.5
    c2 = 0.5
    g1, g2, g3, g4, g5 = pso.init(pop, gen, w, c1, c2, copy2, dummiesLst, createDummies, normalize, 'accuracy')

