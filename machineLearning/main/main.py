import machineLearning.preprocessing.config as cfg

from machineLearning.preprocessing import data
from machineLearning.heuristics import hill, genetic, differential, simulated, swarm, tabu, random,\
    proba, pbil, pbil_diff

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
    elif cfg.general['heuristic'] == 'proba':
        heuristic = proba.Proba(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_pop=cfg.proba['pop'], n_gen=cfg.proba['gen'],
                                            cross_proba=cfg.proba['cross proba'], F=cfg.proba['F'],
                                            low_bound=cfg.proba['low_bound'], up_bound=cfg.proba['up_bound'],
                                            data=copy2, dummiesList=d.dummiesList, createDummies=createDummies,
                                            normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'pbil':
        heuristic = pbil.Pbil(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_pop=cfg.pbil['pop'], n_gen=cfg.pbil['gen'],
                                            learning_rate=cfg.pbil['learning_rate'], mut_proba=cfg.pbil['mut_proba'],
                                            mut_shift=cfg.pbil['mut_shift'], data=copy2, dummiesList=d.dummiesList,
                                            createDummies=createDummies, normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'pbil_diff':
        heuristic = pbil_diff.PbilDiff(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_pop=cfg.pbil_diff['pop'], n_gen=cfg.pbil_diff['gen'],
                                            F=cfg.pbil_diff['F'],
                                            learning_rate=cfg.pbil_diff['learning_rate'],
                                            mut_proba=cfg.pbil_diff['mut_proba'], mut_shift=cfg.pbil_diff['mut_shift'],
                                            data=copy2, dummiesList=d.dummiesList, createDummies=createDummies,
                                            normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'hill':
        heuristic = hill.Hill(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_gen=cfg.hill['gen'], n_neighbors=cfg.hill['nei'],
                                            n_mute_max=cfg.hill['dist'], data=copy2, dummiesList=d.dummiesList,
                                            createDummies=createDummies, normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'tabu':
        heuristic = tabu.Tabu(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_tabu=cfg.tabu['tab'], n_gen=cfg.tabu['gen'], n_neighbors=cfg.tabu['nei'],
                                            n_mute_max=cfg.tabu['dist'], data=copy2, dummiesList=d.dummiesList,
                                            createDummies=createDummies, normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'simulated':
        heuristic = simulated.Simulated(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(temperature=cfg.simulated['temperature'], alpha=cfg.simulated['alpha'],
                                            final_temperature=cfg.simulated['final'], n_mute_max=cfg.simulated['dist'],
                                            data=copy2, dummiesList=d.dummiesList, createDummies=createDummies,
                                            normalize=normalize, metric=metric)
    elif cfg.general['heuristic'] == 'random':
        heuristic = random.Random(d2, d, methods, target, origin, name)
        g1, g2, g3, g4, g5 = heuristic.init(n_gen=cfg.random['gen'], proba=cfg.random['p'],
                                            n_neighbors=cfg.random['nei'], data=copy2,
                                            dummiesList=d.dummiesList, createDummies=createDummies,
                                            normalize=normalize, metric=metric)
    else:
        print(cfg.general['heuristic'] +
              " n'est pas un nom d'heristique correct, veuillez choisir parmi les suivants:\n" +
              "genetic, differential, swarm, proba, pbil, pbil_diff, hill, tabu, simulated, random")
