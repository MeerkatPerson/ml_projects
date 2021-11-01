import numpy as np

import pickle

import matplotlib.pyplot as plt

# Open the file and call pickle.load.
with open("../results/logreg_gridsearch_hyp.txt", "rb") as f:

    results = pickle.load(f)

    # w_initial_dist = normal, batch_size = 10000

    accuracies_te_normal_10000 = np.ones((1,2))

    accuracies_tr_normal_10000 = np.ones((1,2))

    learning_rates_normal_10000 = []

    # w_initial_dist = zeros, batch_size = 10000

    accuracies_te_zeros_10000 = np.ones((1,2))

    accuracies_tr_zeros_10000 = np.ones((1,2))

    learning_rates_zeros_10000 = []

    # w_initial_dist = normal, batch_size = 20000

    accuracies_te_normal_20000 = np.ones((1,2))

    accuracies_tr_normal_20000 = np.ones((1,2))

    learning_rates_normal_20000 = []

    # w_initial_dist = zeros, batch_size = 20000

    accuracies_te_zeros_20000 = np.ones((1,2))

    accuracies_tr_zeros_20000 = np.ones((1,2))

    learning_rates_zeros_20000 = []

    # w_initial_dist = normal, batch_size = 50000

    accuracies_te_normal_50000 = np.ones((1,2))

    accuracies_tr_normal_50000 = np.ones((1,2))

    learning_rates_normal_50000 = []

    # w_initial_dist = zeros, batch_size = 20000

    accuracies_te_zeros_50000 = np.ones((1,2))

    accuracies_tr_zeros_50000 = np.ones((1,2))

    learning_rates_zeros_50000 = []

    # w_initial_dist = normal, batch_size = -1

    accuracies_te_normal_N = np.ones((1,2))

    accuracies_tr_normal_N = np.ones((1,2))

    learning_rates_normal_N= []

    # w_initial_dist = zeros, batch_size = 20000

    accuracies_te_zeros_N = np.ones((1,2))

    accuracies_tr_zeros_N = np.ones((1,2))

    learning_rates_zeros_N = []

    for r in results:
        
        if r['w_initial distr'] == 'normal' and r['batch_size'] == 10000:
        
            # (a) Acc_tr

            accuracy_te_normal_10000 = np.array(r['acc_te']).reshape(1,-1)
            
            acc_te_normal_10000 = np.zeros((1,2))

            acc_te_normal_10000[0,0] = accuracy_te_normal_10000.mean()

            acc_te_normal_10000[0,1] = accuracy_te_normal_10000.std()

            accuracies_te_normal_10000 = np.concatenate((accuracies_te_normal_10000,acc_te_normal_10000), axis = 0)

            # (b) Acc_te

            accuracy_tr_normal_10000 = np.array(r['acc_tr']).reshape(1,-1)
            
            acc_tr_normal_10000 = np.zeros((1,2))

            acc_tr_normal_10000[0,0] = accuracy_tr_normal_10000.mean()

            acc_tr_normal_10000[0,1] = accuracy_tr_normal_10000.std()

            accuracies_tr_normal_10000 = np.concatenate((accuracies_tr_normal_10000,acc_tr_normal_10000), axis = 0)

            # (c) Learning rate

            learning_rates_normal_10000.append(r['learning rate'])

        elif r['w_initial distr'] == 'normal' and r['batch_size'] == 20000:

            accuracy_te_normal_20000 = np.array(r['acc_te']).reshape(1,-1)
            
            acc_te_normal_20000 = np.zeros((1,2))

            acc_te_normal_20000[0,0] = accuracy_te_normal_20000.mean()

            acc_te_normal_20000[0,1] = accuracy_te_normal_20000.std()

            accuracies_te_normal_20000 = np.concatenate((accuracies_te_normal_20000,acc_te_normal_20000), axis = 0)

            # (b) Acc_te

            accuracy_tr_normal_20000 = np.array(r['acc_tr']).reshape(1,-1)
            
            acc_tr_normal_20000 = np.zeros((1,2))

            acc_tr_normal_20000[0,0] = accuracy_tr_normal_20000.mean()

            acc_tr_normal_20000[0,1] = accuracy_tr_normal_20000.std()

            accuracies_tr_normal_20000 = np.concatenate((accuracies_tr_normal_20000,acc_tr_normal_20000), axis = 0)

            # (c) Learning rate

            learning_rates_normal_20000.append(r['learning rate'])

        elif r['w_initial distr'] == 'normal' and r['batch_size'] == 50000:

            accuracy_te_normal_50000 = np.array(r['acc_te']).reshape(1,-1)
            
            acc_te_normal_50000 = np.zeros((1,2))

            acc_te_normal_50000[0,0] = accuracy_te_normal_50000.mean()

            acc_te_normal_50000[0,1] = accuracy_te_normal_50000.std()

            accuracies_te_normal_50000 = np.concatenate((accuracies_te_normal_50000,acc_te_normal_50000), axis = 0)

            # (b) Acc_te

            accuracy_tr_normal_50000 = np.array(r['acc_tr']).reshape(1,-1)
            
            acc_tr_normal_50000 = np.zeros((1,2))

            acc_tr_normal_50000[0,0] = accuracy_tr_normal_50000.mean()

            acc_tr_normal_50000[0,1] = accuracy_tr_normal_50000.std()

            accuracies_tr_normal_50000 = np.concatenate((accuracies_tr_normal_50000,acc_tr_normal_50000), axis = 0)

            # (c) Learning rate

            learning_rates_normal_50000.append(r['learning rate'])

        elif r['w_initial distr'] == 'normal' and r['batch_size'] == -1:

            accuracy_te_normal_N = np.array(r['acc_te']).reshape(1,-1)
            
            acc_te_normal_N = np.zeros((1,2))

            acc_te_normal_N[0,0] = accuracy_te_normal_N.mean()

            acc_te_normal_N[0,1] = accuracy_te_normal_N.std()

            accuracies_te_normal_N = np.concatenate((accuracies_te_normal_N,acc_te_normal_N), axis = 0)

            # (b) Acc_te

            accuracy_tr_normal_N = np.array(r['acc_tr']).reshape(1,-1)
            
            acc_tr_normal_N = np.zeros((1,2))

            acc_tr_normal_N[0,0] = accuracy_tr_normal_N.mean()

            acc_tr_normal_N[0,1] = accuracy_tr_normal_N.std()

            accuracies_tr_normal_N = np.concatenate((accuracies_tr_normal_N,acc_tr_normal_N), axis = 0)

            # (c) Learning rate

            learning_rates_normal_N.append(r['learning rate'])

        elif r['w_initial distr'] == 'zero' and r['batch_size'] == 10000:

            # (a) Acc_tr

            accuracy_te_zeros_10000 = np.array(r['acc_te']).reshape(1,-1)
            
            acc_te_zeros_10000 = np.zeros((1,2))

            acc_te_zeros_10000[0,0] = accuracy_te_zeros_10000.mean()

            acc_te_zeros_10000[0,1] = accuracy_te_zeros_10000.std()

            accuracies_te_zeros_10000 = np.concatenate((accuracies_te_zeros_10000,acc_te_zeros_10000), axis = 0)

            # (b) Acc_te

            accuracy_tr_zeros_10000 = np.array(r['acc_tr']).reshape(1,-1)
            
            acc_tr_zeros_10000 = np.zeros((1,2))

            acc_tr_zeros_10000[0,0] = accuracy_tr_zeros_10000.mean()

            acc_tr_zeros_10000[0,1] = accuracy_tr_zeros_10000.std()

            accuracies_tr_zeros_10000 = np.concatenate((accuracies_tr_zeros_10000,acc_tr_zeros_10000), axis = 0)

            # (c) Learning rate, batch size, 

            learning_rates_zeros_10000.append(r['learning rate'])

        elif r['w_initial distr'] == 'zero' and r['batch_size'] == 20000:

            # (a) Acc_tr

            accuracy_te_zeros_20000 = np.array(r['acc_te']).reshape(1,-1)
            
            acc_te_zeros_20000 = np.zeros((1,2))

            acc_te_zeros_20000[0,0] = accuracy_te_zeros_20000.mean()

            acc_te_zeros_20000[0,1] = accuracy_te_zeros_20000.std()

            accuracies_te_zeros_20000 = np.concatenate((accuracies_te_zeros_20000,acc_te_zeros_20000), axis = 0)

            # (b) Acc_te

            accuracy_tr_zeros_20000 = np.array(r['acc_tr']).reshape(1,-1)
            
            acc_tr_zeros_20000 = np.zeros((1,2))

            acc_tr_zeros_20000[0,0] = accuracy_tr_zeros_20000.mean()

            acc_tr_zeros_20000[0,1] = accuracy_tr_zeros_20000.std()

            accuracies_tr_zeros_20000 = np.concatenate((accuracies_tr_zeros_20000,acc_tr_zeros_20000), axis = 0)

            # (c) Learning rate, batch size, 

            learning_rates_zeros_20000.append(r['learning rate'])

        elif r['w_initial distr'] == 'zero' and r['batch_size'] == 50000:

            # (a) Acc_tr

            accuracy_te_zeros_50000 = np.array(r['acc_te']).reshape(1,-1)
            
            acc_te_zeros_50000 = np.zeros((1,2))

            acc_te_zeros_50000[0,0] = accuracy_te_zeros_50000.mean()

            acc_te_zeros_50000[0,1] = accuracy_te_zeros_50000.std()

            accuracies_te_zeros_50000 = np.concatenate((accuracies_te_zeros_50000,acc_te_zeros_50000), axis = 0)

            # (b) Acc_te

            accuracy_tr_zeros_50000 = np.array(r['acc_tr']).reshape(1,-1)
            
            acc_tr_zeros_50000 = np.zeros((1,2))

            acc_tr_zeros_50000[0,0] = accuracy_tr_zeros_50000.mean()

            acc_tr_zeros_50000[0,1] = accuracy_tr_zeros_50000.std()

            accuracies_tr_zeros_50000 = np.concatenate((accuracies_tr_zeros_50000,acc_tr_zeros_50000), axis = 0)

            # (c) Learning rate, batch size, 

            learning_rates_zeros_50000.append(r['learning rate'])

        else:

            # (a) Acc_tr

            accuracy_te_zeros_N = np.array(r['acc_te']).reshape(1,-1)
            
            acc_te_zeros_N = np.zeros((1,2))

            acc_te_zeros_N[0,0] = accuracy_te_zeros_N.mean()

            acc_te_zeros_N[0,1] = accuracy_te_zeros_N.std()

            accuracies_te_zeros_N = np.concatenate((accuracies_te_zeros_N,acc_te_zeros_N), axis = 0)

            # (b) Acc_te

            accuracy_tr_zeros_N = np.array(r['acc_tr']).reshape(1,-1)
            
            acc_tr_zeros_N = np.zeros((1,2))

            acc_tr_zeros_N[0,0] = accuracy_tr_zeros_N.mean()

            acc_tr_zeros_N[0,1] = accuracy_tr_zeros_N.std()

            accuracies_tr_zeros_N = np.concatenate((accuracies_tr_zeros_N,acc_tr_zeros_N), axis = 0)

            # (c) Learning rate, batch size, 

            learning_rates_zeros_N.append(r['learning rate'])


    # Grab means

    # batch_size 10 000

    accuracies_te_normal_10000 = accuracies_te_normal_10000[1:]

    accuracies_te_zeros_10000 = accuracies_te_zeros_10000[1:]

    accuracies_te_normal_mean_10000 = accuracies_te_normal_10000[:,0]

    accuracies_te_zeros_mean_10000 = accuracies_te_zeros_10000[:,0]

    # batch_size 20 000

    accuracies_te_normal_20000 = accuracies_te_normal_20000[1:]

    accuracies_te_zeros_20000 = accuracies_te_zeros_20000[1:]

    accuracies_te_normal_mean_20000 = accuracies_te_normal_20000[:,0]

    accuracies_te_zeros_mean_20000 = accuracies_te_zeros_20000[:,0]

    # batch_size 50 000

    accuracies_te_normal_50000 = accuracies_te_normal_50000[1:]

    accuracies_te_zeros_50000 = accuracies_te_zeros_50000[1:]

    accuracies_te_normal_mean_50000 = accuracies_te_normal_50000[:,0]

    accuracies_te_zeros_mean_50000 = accuracies_te_zeros_50000[:,0]

    # batch_size N

    accuracies_te_normal_N = accuracies_te_normal_N[1:]

    accuracies_te_zeros_N = accuracies_te_zeros_N[1:]

    accuracies_te_normal_mean_N = accuracies_te_normal_N[:,0]

    accuracies_te_zeros_mean_N = accuracies_te_zeros_N[:,0]

    # Transform lists to arrays

    learning_rates_normal_10000 = np.array(learning_rates_normal_10000)

    learning_rates_normal_20000 = np.array(learning_rates_normal_20000)

    learning_rates_normal_50000 = np.array(learning_rates_normal_50000)

    learning_rates_normal_N = np.array(learning_rates_normal_N)

    learning_rates_zeros_10000 = np.array(learning_rates_zeros_10000)

    learning_rates_zeros_20000 = np.array(learning_rates_zeros_20000)

    learning_rates_zeros_50000 = np.array(learning_rates_zeros_50000)

    learning_rates_zeros_N = np.array(learning_rates_zeros_N)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.style.use('seaborn') # need some A E S T H E T I C S

    # Plot separately for normal/zeros w_initial

    scatter_normal_10000 = plt.scatter(learning_rates_normal_10000, accuracies_te_normal_mean_10000, c="#7FFFD4", marker="o", edgecolor = 'black', linewidth=1, alpha=0.5, s = 100)

    scatter_normal_20000 = plt.scatter(learning_rates_normal_20000, accuracies_te_normal_mean_20000, c="#5F9EA0", marker="o", edgecolor = 'black', linewidth=1, alpha=0.5, s = 100)

    scatter_normal_50000 = plt.scatter(learning_rates_normal_50000, accuracies_te_normal_mean_50000, c="#6495ED", marker="o", edgecolor = 'black', linewidth=1, alpha=0.5, s = 100)

    scatter_normal_N = plt.scatter(learning_rates_normal_N, accuracies_te_normal_mean_N, c="#008B8B", marker="o", edgecolor = 'black', linewidth=1, alpha=0.5, s = 100)

    scatter_zeros_10000 = plt.scatter(learning_rates_zeros_10000, accuracies_te_zeros_mean_10000, c="#7FFFD4", marker="s", edgecolor = 'black', linewidth=1, alpha=0.5, s = 100)

    scatter_zeros_20000 = plt.scatter(learning_rates_zeros_20000, accuracies_te_zeros_mean_20000, c="#5F9EA0", marker="s", edgecolor = 'black', linewidth=1, alpha=0.5, s = 100)

    scatter_zeros_50000 = plt.scatter(learning_rates_zeros_50000, accuracies_te_zeros_mean_50000, c="#6495ED", marker="s", edgecolor = 'black', linewidth=1, alpha=0.5, s = 100)

    scatter_zeros_N = plt.scatter(learning_rates_zeros_N, accuracies_te_zeros_mean_N, c="#008B8B", marker="s", edgecolor = 'black', linewidth=1, alpha=0.5, s = 100)

    plt.xscale('log')

    plt.xlabel('Learning Rate')

    plt.ylabel('Test accuracy')

    plt.legend((scatter_normal_10000,scatter_normal_20000,scatter_normal_50000,scatter_normal_N,scatter_zeros_10000,scatter_zeros_20000,scatter_zeros_50000,scatter_zeros_N), 
                ('Normal 10000','Normal 20000','Normal 50000','Normal N','Zero 10000','Zero 20000','Zero 50000','Zero N'),
                bbox_to_anchor=(1,0),
                loc="lower left", 
                ncol=1,
                title="w_initial dist/batch size")

    plt.tight_layout()

    plt.show()