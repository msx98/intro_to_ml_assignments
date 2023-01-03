import backprop_data

import backprop_network

# Question (b)
"""
Train a one-hidden layer neural network as in the example given above
(e.g., training set of size 10000, one hidden layer of size 40). For each learning rate
in {0.001, 0.01, 0.1, 1, 10, 100}, plot the training accuracy, training loss (ℓ(W)) and test
accuracy across epochs (3 plots: each contains the curves for all learning rates). For the
test accuracy you can use the one label accuracy function, for the training accuracy use
the one hot accuracy function and for the training loss you can use the loss function.
All functions are in the Network class.
The test accuracy with leaning rate 0.1 in the final epoch should be above 80%.
What happens when the learning rate is too small or too large? Explain the phenomenon
"""
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('headless')
    except:
        pass
import matplotlib.pyplot as plt
import numpy as np
plt.ioff()
def qb():
    print(f"Running qb()")
    training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
    learning_rates = [0.001, 0.01]#, 0.1, 1, 10, 100]
    results = dict()
    results['training_loss'] = []
    results['training_accuracy'] = []
    results['test_accuracy'] = []
    for learning_rate in learning_rates:
        net = backprop_network.Network([784, 40, 10])
        net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=learning_rate, test_data=test_data)
        results['training_loss'].append(net.training_loss)
        results['training_accuracy'].append(net.training_accuracy)
        results['test_accuracy'].append(net.test_accuracy)
        print(f"Done, learning rate: {learning_rate}")
    
    def plot(selection:str):
        str_name = selection.replace("_"," ")
        ax = {True: plt, False: plt}
        for skip_first in [True, False]:
            plt.figure()
            plt.title(f"{str_name} vs epoch " + ("including initial" if not skip_first else "without first datapoint"))
            data = [x[1:] for x in results[selection]] if skip_first else results[selection]
            for i, learning_rate in enumerate(learning_rates):
                xrange = np.arange(len(data[i])) - (1 if not skip_first else 0) # [-1, ..., max_epoch]
                plt.plot(xrange, data[i], label=f"lr = {learning_rate}")
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel(str_name)
            with_str = "with" if not skip_first else "without"
            plt.savefig(f"{selection}_{with_str}.png", dpi=600)
    
    plot('training_loss')
    plot('training_accuracy')
    plot('test_accuracy')
    print(f"Done with qb()")


def qc():
    print(f"Running qc()")
    training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.skip_training_loss = True
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)
    test_accuracy = net.one_label_accuracy(test_data)
    print(f"qc: test_accuracy = {test_accuracy}")
    with open("qc.txt", "w") as f:
        f.write(f"qc: test_accuracy = {test_accuracy}")
    print(f"Done with qc()")


qb()
qc()

