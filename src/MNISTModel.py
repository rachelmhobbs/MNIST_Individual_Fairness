import tensorflow as tf
import numpy as np
import tf_helper_functions as tf_helper

class MNISTModel:

    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.config["inputs"]])

        with tf.name_scope("dnn_model"):

            self.w1 = tf_helper.weights([self.config["inputs"], self.config["hidden1"]], name="w1")
            b1 = tf_helper.bias([self.config["hidden1"]], name="b1")
            z1 = tf_helper.dense_layer(self.x, self.w1, b1, activation=tf.nn.relu)

            self.w2 = tf_helper.weights([self.config["hidden1"], self.config["hidden2"]], name="w2")
            b2 = tf_helper.bias([self.config["hidden2"]], name="b2")
            z2 = tf_helper.dense_layer(z1, self.w2, b2, activation=tf.nn.relu)

            self.w3 = tf_helper.weights([self.config["hidden2"], self.config["outputs"]], name="w3")
            b3 = tf_helper.bias([self.config["outputs"]], name="b3")
            z3 = tf_helper.dense_layer(z2, self.w3, b3, activation=None)

        with tf.name_scope("loss"):
            self.y = tf.placeholder(dtype=tf.int32, shape=None)
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=z3)
            self.loss = tf.reduce_mean(self.xentropy, name="loss")

            self.softmax = tf.nn.softmax(z3)
            self.prediction_perc, predictions = tf.nn.top_k(self.softmax, k=1, name="predictions")
            self.predictions = tf.reshape(tf.cast(predictions, tf.int32), shape=[-1])

        with tf.name_scope("lipshitz_regularization"):
            self.w1_projection, self.w1_norm = tf_helper.lipschitz_projection(self.w1, p_norm=self.config["p_norm"], lamda=self.config["lamda1"])
            self.w2_projection, self.w2_norm = tf_helper.lipschitz_projection(self.w2, p_norm=self.config["p_norm"], lamda=self.config["lamda2"])
            self.w3_projection, self.w3_norm = tf_helper.lipschitz_projection(self.w3, p_norm=self.config["p_norm"], lamda=self.config["lamda3"])

        with tf.name_scope("training_op"):
            self.learning_rate = tf.Variable(self.config["learning_rate"])
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.training_op = optimizer.minimize(self.loss)

        with tf.name_scope("accuracy"):
            self.correct_predicitions = tf.nn.in_top_k(z3, self.y, 1)
            #self.accuracy = tf.reduce_mean(tf.cast(self.correct_predicitions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.y), tf.float32))

        with tf.name_scope("class_accuracy"):
            self.number_class = tf.placeholder(dtype=tf.int32, shape=None)
            self.class_labels = tf.equal(self.number_class, self.y)
            self.class_predictions = tf.equal(self.number_class, self.predictions)
            self.class_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.class_labels, self.class_predictions), tf.float32))

        with tf.name_scope("confusion_matrix"):
            self.confusion_matrix = tf.confusion_matrix(labels=self.y, predictions=self.predictions, num_classes=self.config["outputs"], name="confusion_matrix")
