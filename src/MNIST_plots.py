import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os

def plot_metrics(metrics_data, config, display=False):

    '''
    Build plots for metrics_data and save plots to model_dir specified in config

    Parameters:
        metrics_data: metrics from training
        config: configuration dictionary for network parameters
        display: Boolean for if plots should be displayed during runtime or not. Default False.
    '''
    pp = PdfPages(os.path.join(config["model_dir"],"graphs_2.pdf"))

    #Plot loss
    min_test_loss = min(metrics_data["test_loss"])
    plt.figure(1)
    plt.grid(True)
    plt.title("Cross Entropy Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(metrics_data["test_loss"], label="Test")
    plt.plot(metrics_data["train_loss"], label="Train")
    plt.hlines(min_test_loss, xmin=0, xmax=config["num_epochs"], colors="r", linestyle="dashed", label="Min. Test Cost="+str(min_test_loss))
    plt.legend()
    pp.savefig()

    #Plot accuracy data
    plt.figure(2)
    plt.grid(True)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(metrics_data["test_acc"], label="Test_Acc.")
    plt.plot(metrics_data["train_acc"], label="Train_Acc.")
    plt.legend()
    pp.savefig()

    #plot weight norms data
    plt.figure(3)
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("l"+str(config["p_norm"])+" norm")
    plt.plot(metrics_data["w1_norm"], label="w1_norm")
    plt.hlines(config["lamda1"], xmin=0, xmax=config["num_epochs"], colors="m", linestyle='dashed', linewidth=4, label="Lamda1="+str(config["lamda1"]))
    plt.plot(metrics_data["w2_norm"], label="w2_norm")
    plt.hlines(config["lamda2"], xmin=0, xmax=config["num_epochs"], colors="k", linestyle='dashed', linewidth=4, label="Lamda2="+str(config["lamda2"]))
    plt.plot(metrics_data["w3_norm"], label="w3_norm")
    plt.hlines(config["lamda3"], xmin=0, xmax=config["num_epochs"], colors="r", linestyle='dashed', linewidth=4, label="Lamda3="+str(config["lamda3"]))
    plt.legend()
    pp.savefig()

    #plot individual class Accuracy
    plt.figure(4)
    plt.grid(True)
    plt.title("Individual Class Accuracy")
    plt.ylim(ymin=0, ymax=1.0)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    for i in range(config["outputs"]):
        plt.plot(metrics_data["class_{}_acc".format(i)], label="Class_{}".format(i))
    plt.legend()
    pp.savefig()

    #plot confusion matrix as matrix image
    plt.figure(5)
    plt.title("Confusion Matrix")
    plt.matshow(metrics_data["confusion_matrix"], cmap=plt.cm.jet)
    cbr = plt.colorbar()
    cbr.set_label("Number of Predictions Mapped to Label")
    plt.xlabel("Predictions")
    plt.ylabel("Labels")
    pp.savefig()

    pp.close()
    if(display):
        plt.show()
