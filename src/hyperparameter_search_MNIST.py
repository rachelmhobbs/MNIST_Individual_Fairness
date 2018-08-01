import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from MNISTModel import MNISTModel
import MNIST_plots

def feed_single_class(X, y, clas, num_classes):
    '''
    Returns only a batch containing the specified class/category

    Parameters:
        X: Input data as numpy array
        y: Labels for Input data(also numpy array)
        clas: The class or category to keep
        num_classes: The number of classes/categories in entire label set
    Returns:
        X_batch: New Input data containing only data of given class/category
        y_batch: New Label/output data containing only data of given class/category
    '''
    X_batch = np.array(X)
    y_batch = np.array(y)
    for i in range(num_classes):
        if(i != clas):
            X_batch, y_batch, _, _ = remove_class(X_batch, y_batch, i, 1.0)
    return X_batch, y_batch

def remove_class(X, y, clas, percentage=1.0):
    '''
    Removes a percentage of a class of data in a given dataset

    Parameters:
        X: Input data as numpy array
        y: Labels for Input data(also numpy array)
        clas: The class/category to keep
        percentage: Default = 1.0. The precentage of data from a given class to be removed from dataset
    Returns:
        X_batch: New input data with percentage of a give class/category removed
        y_batch: New label/output data same corresponding data removed as input data
        number_of_instance_deleted
        number_of_class: The total number of a given category in the input data set before being removed.
    '''
    indicies =  np.reshape(np.where(y==clas), [-1])
    np.random.shuffle(indicies)
    number_of_class = indicies.shape[0]
    number_of_instances_deleted = int(percentage*number_of_class)
    X_batch = np.delete(X, indicies[:number_of_instances_deleted], axis=0)
    y_batch = np.delete(y, indicies[:number_of_instances_deleted])

    return X_batch, y_batch, number_of_instances_deleted, number_of_class

def train(config):
    '''
    Trains the MNIST dataset on the MNISTModel DNN model and evalutes metrics such as
    overall accuracy, class accuracy/precision/recall/f1_score, individual image distances,
    layer weight norms, etc

    Parameters:
        config: A dictionary containing all the necessary configuration parameters for the network
                and training
    Returns:
        metrics: A dictionary containing a wide variety of metrics gathered over the entire training
                process
    '''
    metrics = {"w1_norm":[], "w2_norm":[], "w3_norm":[], "loss_val":[], "train_loss":[], "test_loss":[], "train_acc":[], "test_acc":[]}
    for i in range(config["outputs"]):
        metrics["class_{}_acc".format(i)] = [] #accuracy
        metrics["class_{}_prec".format(i)] = [] #precision
        metrics["class_{}_rec".format(i)] = [] #recall
        metrics["class_{}_f1".format(i)] = []

    mnist = input_data.read_data_sets("../data")
    model = MNISTModel(config) #configure MNIST DNN model

    num_batches = mnist.train.num_examples // config["batch_size"]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for epoch in range(config["num_epochs"]):
            #reset local variables for class precision
            '''
            sess.run([model.precision_vars_init[c] for c in range(config["outputs"])])
            sess.run([model.recall_vars_init[k] for k in range(config["outputs"])])
            '''
            train_loss_accum = 0
            train_acc_accum = 0

            #mini-batch gradient descent
            for iteration in range(num_batches):
                X_batch, y_batch = mnist.train.next_batch(config["batch_size"])

                #X_batch, y_batch, _, _= remove_class(X_batch, y_batch, 5, percentage=.99)
                X_batch, y_batch, _, _ = remove_class(X_batch, y_batch, 3, percentage=.99)

                training_dict = {model.x:X_batch, model.y:y_batch}
                sess.run(model.training_op, feed_dict=training_dict)
                #sess.run([model.w1_projection, model.w2_projection, model.w3_projection])
                train_loss_accum+=sess.run(model.loss, feed_dict=training_dict)
                train_acc_accum+= sess.run(model.accuracy, feed_dict=training_dict)

            #compute the specified "norm" for each weight matrix in DNN
            metrics["w1_norm"].append(sess.run(model.w1_norm))
            metrics["w2_norm"].append(sess.run(model.w2_norm))
            metrics["w3_norm"].append(sess.run(model.w3_norm))

            #compute training & testing accuracy/loss
            metrics["train_acc"].append(train_acc_accum/num_batches)
            metrics["train_loss"].append(train_loss_accum/num_batches)
            metrics["test_acc"].append(sess.run(model.accuracy, feed_dict={model.x:mnist.validation.images, model.y:mnist.validation.labels}))
            metrics["test_loss"].append(sess.run(model.loss, feed_dict={model.x:mnist.validation.images, model.y:mnist.validation.labels}))

            for i in range(config["outputs"]):
                #compute single class accuracy, precision, recall, and f1 score
                X_batch, y_batch = feed_single_class(mnist.validation.images, mnist.validation.labels, i, config["outputs"])
                metrics["class_{}_acc".format(i)].append(sess.run(model.accuracy,  feed_dict={model.number_class:i, model.x:X_batch, model.y:y_batch}))
                print("Class", i, "Accuracy:", metrics["class_{}_acc".format(i)][-1])

            print("Test Accuracy:", metrics["test_acc"][-1])
            print("Training Progress {:2.1%}".format(float((epoch+1)/config["num_epochs"])), end="\n", flush=True)
        
        metrics["confusion_matrix"] = sess.run(model.confusion_matrix, feed_dict={model.x:mnist.validation.images, model.y:mnist.validation.labels})
    MNIST_plots.plot_metrics(metrics, config, display=False)
    return(metrics)
if __name__ == "__main__":
    config = {"inputs":28*28, "hidden1":300, "hidden2":100, "outputs":10, "p_norm":np.inf, "lamda1":0.4, "lamda2":0.4, "lamda3":0.4,
                "learning_rate":0.01, "batch_size":124, "num_epochs":10, "model_dir":"C:\Machine_Learning\ML Projects\Fairness\MNIST_Individual_Fairness\data_aquisition"}
    train(config)
