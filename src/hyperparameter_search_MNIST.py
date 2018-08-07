import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from MNISTModel import MNISTModel
import MNIST_plots
import os
from datetime import datetime

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

def distance_computation(data, p_norm):
    '''
    '''
    dists = np.zeros((data.shape[0], data.shape[0]))

    #broadcasting to do single loop distance computation
    for i in range(data.shape[0]):
        dists[i, :] = np.linalg.norm(data-data[i, :], ord=p_norm, axis=1)
    return(dists)

def prediction_mapping(label_data):
    '''
    '''

    prediction_map = np.zeros((label_data.shape[0], label_data.shape[0]))

    for i in range(label_data.shape[0]):
        prediction_map[i, :] = np.reshape(label_data == label_data[i, :], (label_data.shape[0]))

    return(prediction_map)


def train(config, mnist):
    '''
    Trains the MNIST dataset on the MNISTModel DNN model and evalutes metrics such as
    overall accuracy, class accuracy/precision/recall/f1_score, individual image distances,
    layer weight norms, etc

    Parameters:
        config: A dictionary containing all the necessary configuration parameters for the network
                and training
        mnist: The read in mnist dataset
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

    #reset all tf graphs to reassure no old graphs are still alive
    tf.reset_default_graph()
    if not mnist:
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


                for number in config["removed_classes"]:
                    X_batch, y_batch, _, _ = remove_class(X_batch, y_batch, number, percentage=config["removed_perc"])

                training_dict = {model.x:X_batch, model.y:y_batch}
                sess.run(model.training_op, feed_dict=training_dict)    #train model

                if(config["lipschitz_constraint"] == True):
                    sess.run([model.w1_projection, model.w2_projection, model.w3_projection])

                #calculate runtime loss and accuracy for training set
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

        #save tensorflow model
        model.save_model(sess, config["model_dir"])

        #compute distances matrices for each layer
        input_data = {model.x:mnist.validation.images[:20], model.y:mnist.validation.labels[:20]}
        val_layer1_output = sess.run(model.z1, feed_dict=input_data)
        val_layer2_output = sess.run(model.z2, feed_dict=input_data)
        val_layer3_output = sess.run(model.z3, feed_dict=input_data) #layer 3 is befor activation
        val_softmax_output = sess.run(model.softmax, feed_dict=input_data)

        metrics["val_input_data_dists"] = distance_computation(np.array(mnist.validation.images[:20]), p_norm=config["p_norm"])
        metrics["val_input_data_dists_max"] = np.amax(metrics["val_input_data_dists"])
        metrics["val_layer1_dists"] = distance_computation(val_layer1_output, p_norm=config["p_norm"])
        metrics["val_layer1_dists_max"] = np.amax(metrics["val_layer1_dists"])
        metrics["val_layer2_dists"] = distance_computation(val_layer2_output, p_norm=config["p_norm"])
        metrics["val_layer2_dists_max"] = np.amax(metrics["val_layer2_dists"])
        metrics["val_layer3_dists"] = distance_computation(val_layer3_output, p_norm=config["p_norm"])
        metrics["val_layer3_dists_max"] = np.amax(metrics["val_layer3_dists"])
        metrics["val_softmax_output"] = distance_computation(val_softmax_output, p_norm=config["p_norm"])
        metrics["val_softmax_output_max"] = np.amax(metrics["val_softmax_output"])
        metrics["val_pred_map"] = prediction_mapping(
                                    np.reshape(np.array(mnist.validation.labels), (-1, 1))[:20])

    MNIST_plots.plot_metrics(metrics, config, display=False)
    return(metrics)

def hyperparameter_train_constant_lamdas(config, hyperparameters, test_dir):
    '''
    Train biased mnist with multiple lamda values. Save all the data to test_dir.
    '''
    #extract dataset
    mnist = input_data.read_data_sets("../data")

    #First train without lipschitz constraints
    config["lipschitz_constraint"] = False

    save_dir = os.path.join(test_dir, "original")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    config["model_dir"] = save_dir
    metrics = train(config, mnist)
    with open(os.path.join(config["model_dir"], "model_metrics.txt"), "w") as metrics_file:
        metrics_file.write(str(metrics))
    with open(os.path.join(config["model_dir"], "model_config.txt"), "w") as config_file:
        config_file.write(str(config))

    param_keys = ["lamda1", "lamda2", "lamda3"]
    del metrics

    #train with lipschitz_constraints
    config["lipschitz_constraint"] = True
    for hyperparam in hyperparameters:

        print("\n\nConstant Lamdas Training. Current Lamda Val:", hyperparam, "\n\n")

        #update config with new hyperparameters
        for keys in param_keys:
            config[keys] = hyperparam

        save_dir = os.path.join(test_dir, "const_lamdas_val_"+str(hyperparam))
        #if directory does not exist, create it
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        config["model_dir"] = save_dir

        #train model
        metrics = train(config, mnist)
        with open(os.path.join(config["model_dir"], "model_metrics.txt"), "w") as metrics_file:
            metrics_file.write(str(metrics))
        with open(os.path.join(config["model_dir"], "model_config.txt"), "w") as config_file:
            config_file.write(str(config))
        del metrics


if __name__ == "__main__":
    config = {"inputs":28*28, "hidden1":300, "hidden2":100, "outputs":10, "p_norm":np.inf, "lipschitz_constraint":False, "lamda1":0.25, "lamda2":0.25, "lamda3":0.25,
                "learning_rate":0.01, "batch_size":124, "num_epochs":1, "removed_classes":[4], "removed_perc": 0.95, "model_dir":"../data_acquisition",
                "graph_pdf_file":"graphs.pdf"}
    lamda_hyperparams = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    time_now = datetime.now()
    hyperparameter_train_constant_lamdas(config, lamda_hyperparams, "C:\Machine_Learning\ML Projects\Fairness\MNIST_Individual_Fairness\data_acquisition\\test_" + time_now.strftime("%Y_%m_%d_%H_%M_%S"))
