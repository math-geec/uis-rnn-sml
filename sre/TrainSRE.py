import tensorflow as tf
import numpy as np
import os
import sys
import time
from utils import random_batch, normalize, similarity, loss_cal, optim
from configuration import get_config
from tensorflow.contrib import rnn


def train(path):
    tf.reset_default_graph()  # reset graph

    # draw graph
    batch = tf.placeholder(shape=[None, config.N * config.M, 20],
                           dtype=tf.float32)  # input batch (time x batch x n_mel)
    lr = tf.placeholder(dtype=tf.float32)  # learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False)
    w = tf.get_variable("w", initializer=np.array([10], dtype=np.float32))
    b = tf.get_variable("b", initializer=np.array([-5], dtype=np.float32))

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(os.path.join(config.model_path, 'logfile.log'), "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            # this handles the flush command by doing nothing.
            # you might want to specify some extra behavior here.
            pass

            # embedding lstm (3-layer default)

    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in
                      range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)  # define lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32,
                                       time_major=True)  # for TI-VS must use dynamic rnn
        embedded = outputs[-1]  # the last ouput is the embedded d-vector
        embedded = normalize(embedded)  # normalize
    print("embedded size: ", embedded.shape)

    # loss
    sim_matrix = similarity(embedded, w, b)
    print("similarity matrix size: ", sim_matrix.shape)
    loss = loss_cal(sim_matrix, type=config.loss)

    # optimizer operation
    trainable_vars = tf.trainable_variables()  # get variable list
    optimizer = optim(lr)  # get optimizer (type is determined by configuration)
    grads, vars = zip(*optimizer.compute_gradients(loss))  # compute gradients of variables with respect to loss
    grads_clip, _ = tf.clip_by_global_norm(grads, 3.0)  # l2 norm clipping by 3
    grads_rescale = [0.01 * grad for grad in grads_clip[:2]] + grads_clip[2:]  # smaller gradient scale for w, b
    train_op = optimizer.apply_gradients(zip(grads_rescale, vars), global_step=global_step)  # gradient update operation

    # check variables memory
    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables :", variable_count)

    # record loss
    loss_summary = tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    # training session
    with tf.Session() as sess:
        if not config.restore:
            tf.global_variables_initializer().run()
            os.makedirs(os.path.join(path, "Check_Point"), exist_ok=True)  # make folder to save model
            os.makedirs(os.path.join(path, "logs"), exist_ok=True)  # make folder to save log
        writer = tf.summary.FileWriter(os.path.join(path, "logs"), sess.graph)
        epoch = 0
        lr_factor = 1  # lr decay factor ( 1/2 per 10000 iteration)
        loss_acc = 0  # accumulated loss ( for running average of loss)

        sys.stdout = Logger()

        if config.restore:
            # saver = tf.train.import_meta_graph(config.restore_model)
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(config.model_path, "Check_Point")))
        for iter in range(config.iteration):
            # run forward and backward propagation and update parameters
            _, loss_cur, summary = sess.run([train_op, loss, merged],
                                            feed_dict={batch: random_batch(), lr: config.lr * lr_factor})

            loss_acc += loss_cur  # accumulated loss for each 100 iteration

            if iter % 10 == 0:
                writer.add_summary(summary, iter)  # write at tensorboard
            if (iter + 1) % 100 == 0:
                print("(iter : %d) loss: %.4f" % ((iter + 1), loss_acc / 100))
                loss_acc = 0  # reset accumulated loss
            if (iter + 1) % 10000 == 0:
                lr_factor *= 0.8  # lr decay
                print("learning rate is decayed! current lr : ", config.lr * lr_factor)
            if (iter + 1) % 10000 == 0:
                saver.save(sess, os.path.join(path, "./Check_Point/model.ckpt"), global_step=iter // 10000)
                print("model is saved!")


config = get_config()
tf.reset_default_graph()
# start training
if not config.restore:
    os.makedirs(config.model_path)
train(config.model_path)
