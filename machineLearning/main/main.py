import machineLearning.preprocessing.config as cfg

from machineLearning.preprocessing import data
from machineLearning.heuristics import hill, genetic, iterated, differential, simulated, swarm, tabu, vns, ant

if __name__ == '__main__':
    d = data.Data(name=cfg.general['dataset'], target=cfg.general['target'], dropColsList=cfg.general['dropcol'],
                  dropClassList=cfg.general['dropclass'])
    d2, target, copy, copy2, copy3, copy4, dummiesLst, ratio, chi2, anova, origin =\
        d.ready(deleteCols=cfg.general['deletecol'], dropna=cfg.general['dropna'],
                thresholdDrop=cfg.general['thresholddrop'], createDummies=cfg.general['createdummies'],
                normalize=cfg.general['normalize'])
    methods = cfg.general['method']
    name = cfg.general['dataset']
    createDummies = cfg.general['createdummies']
    normalize = cfg.general['normalize']
    metric = cfg.general['metric']
    if cfg.general['heuristic'] == 'genetic':
        heuristic = genetic.Genetic(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_pop=cfg.genetic['pop'], n_gen=cfg.genetic['gen'],
                                            n_mut=cfg.genetic['mut'], data=copy2, dummiesList=d.dummiesList,
                                            createDummies=createDummies, normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'differential':
        heuristic = differential.Differential(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_pop=cfg.differential['pop'], n_gen=cfg.differential['gen'],
                                            cross_proba=cfg.differential['cross proba'], F=cfg.differential['F'],
                                            data=copy2, dummiesList=d.dummiesList, createDummies=createDummies,
                                            normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'swarm':
        heuristic = swarm.PSO(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_pop=cfg.swarm['pop'], n_gen=cfg.swarm['gen'], w=cfg.swarm['w'],
                                            c1=cfg.swarm['c1'], c2=cfg.swarm['c2'], data=copy2,
                                            dummiesList=dummiesLst, createDummies=createDummies, normalize=normalize,
                                            metric=metric)
    elif cfg.general['heuristic'] == 'ant':
        heuristic = ant.ACO(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_pop=cfg.ant['pop'], n_gen=cfg.ant['gen'], p=cfg.ant['p'],
                                            phi=cfg.ant['phi'], alpha=cfg.ant['alpha'], data=copy2,
                                            dummiesList=dummiesLst, createDummies=createDummies, normalize=normalize,
                                            metric=metric)
    elif cfg.general['heuristic'] == 'hill':
        heuristic = hill.Hill(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_gen=cfg.hill['gen'], n_neighbors=cfg.hill['nei'],
                                            n_mute_max=copy.columns.size - 1, data=copy2, dummiesList=d.dummiesList,
                                            createDummies=createDummies, normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'tabu':
        heuristic = tabu.Tabu(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_tabu=cfg.tabu['tab'], n_gen=cfg.tabu['gen'], n_neighbors=cfg.tabu['nei'],
                                            n_mute_max=copy.columns.size - 1, data=copy2, dummiesList=d.dummiesList,
                                            createDummies=createDummies, normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'simulated':
        heuristic = simulated.Simulated(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(temperature=cfg.simulated['temperature'], alpha=cfg.simulated['alpha'],
                                            final_temperature=cfg.simulated['final'], data=copy2,
                                            dummiesList=d.dummiesList, createDummies=createDummies,
                                            normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'vns':
        heuristic = vns.VNS(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_gen=cfg.vns['gen'], n_gen_vnd=cfg.vns['gen vnd'],
                                            n_neighbors=cfg.vns['nei'], kmax=cfg.vns['kmax'], data=copy2,
                                            dummiesList=d.dummiesList, createDummies=createDummies,
                                            normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'iterated':
        heuristic = iterated.Iterated(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_gen=cfg.iterated['gen'], n_gen_vnd=cfg.iterated['gen vnd'],
                                            n_neighbors=cfg.iterated['nei'], kmax=cfg.iterated['kmax'], data=copy2,
                                            dummiesList=d.dummiesList, createDummies=createDummies,
                                            normalize=normalize, metric=metric)
    else:
        print(cfg.general['heuristic'] +
              " n'est pas un nom d'heristique correct, veuillez choisir parmis les suivants:\n" +
              "genetic, differential, swarm, ant, hill, tabu, simulated, vns, iterated")
