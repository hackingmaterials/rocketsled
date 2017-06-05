from matplotlib import pyplot
import pickle

def addtofig(ulm, length, color, label):
    upper = ulm['upper'][0:length]
    lower = ulm['lower'][0:length]
    mean = ulm['mean'][0:length]
    x = range(len(mean))
    pyplot.plot(x, mean, color=color, linewidth=2.5, label=label)
    pyplot.plot(x, lower, color=color, linewidth=0.5, alpha=0.3)
    pyplot.plot(x, upper, color=color, linewidth=0.5, alpha=0.3)
    pyplot.fill_between(x, lower, upper, color=color, alpha=0.13)

def depickle(file):
    return pickle.load(open(file, 'rb'))

if __name__=="__main__":
    rf_withz = depickle('RF_withz.p')
    gb_withz = depickle('GB_withz.p')
    br_withz = depickle('BR_withz.p')
    ab_withz = depickle('AB_withz.p')
    rf_noz = depickle('RF_noz.p')
    gb_noz = depickle('GB_noz.p')
    br_noz = depickle('BR_noz.p')
    ab_noz = depickle('AB_noz.p')
    random_guess = depickle('random_guessing.p')

    # per_rf = depickle('perovskites_RandomForestRegressor_noz_500iters_5runs.p')
    # per_ran = depickle('perovskites_random_guess_noz_500iters_5runs.p')

    # addtofig(per_rf, 500, 'royalblue', 'Random Forest no z')
    # addtofig(per_ran, 500, 'red', 'Random guessing')


    # addtofig(rf_withz, 30, 'dodgerblue', 'RF with z')
    # addtofig(gb_withz, 30, 'goldenrod', 'GB Trees with z')
    # addtofig(br_withz, 30, 'forestgreen', 'Bagging with z')
    addtofig(ab_withz, 30, 'darkorchid', 'Adaboost with z')
    # addtofig(rf_noz, 30, 'lightslategrey', 'RF without z')
    # addtofig(gb_noz, 30, 'khaki', 'GB Trees without z')
    # addtofig(br_noz, 30, 'lightgreen', 'Bagging without z')
    addtofig(ab_noz, 30, 'plum', 'Adaboost without z')
    addtofig(random_guess, 30, 'red', 'Random')

    pyplot.legend(loc='upper right', prop={'size':8})
    pyplot.xlabel("Iteration")
    pyplot.ylabel("Best value")
    # pyplot.title("{} Predictor Performance".format(predictor))
    pyplot.savefig('RFcomparison.png')
    pyplot.show()