# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
from prepareLSTM import DataLoad

n_input = 220
n_steps = 9  # time steps
n_hidden = 128  # hidden layer num of features
n_classes = 16  # Indian Pine total classes (0-15 digits) //共十六大类

def rnn_model(x, weights, biases):
	"""RNN (LSTM or GRU) model for image"""
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(x, n_steps, 0)
	"""inputs: A length T list of inputs, each a `Tensor` of shape
      `[batch_size, input_size]`, or a nested tuple of such elements."""

	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights) + biases

def train():
	"""Train an image classifier"""
	"""Step 0: load image data and training parameters"""
	parameter_file = 'parameters.json'
	params = json.loads(open(parameter_file).read())
	train_loader, test_loader = DataLoad(params['batch_size'])

	"""Step 1: build a rnn model for image"""
	x = tf.placeholder("float", [None, n_steps, n_input])
	y = tf.placeholder("float", [None, n_classes])

	weights = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights')
	biases = tf.Variable(tf.random_normal([n_classes]), name='biases')

	pred = rnn_model(x, weights, biases)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(cost)

	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	# 计算混淆矩阵
	calc_confusion = tf.confusion_matrix(tf.argmax(y,1), tf.argmax(pred,1))

	"""Step 2: train the image classification model"""
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		Epoch = 5

		"""Step 2.0: create a directory for saving model files"""
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_pavia_model_" + timestamp))
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.all_variables())

		"""Step 2.1: train the image classifier batch by batch"""
		for i in range(Epoch):
			print('Epoch {:1d} starts: '.format(i+1))
			for iteration, (batch_x, batch_y) in enumerate(train_loader):
				sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

				"""Step 2.2: save the model"""
				if iteration % params['display_step'] == 0:
					path = saver.save(sess, checkpoint_prefix, global_step=iteration)
					acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
					loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
					print('Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}'.format(iteration * params['batch_size'], loss, acc))
		print("The training is done")

		"""Step 3: test the model"""
		test_len = 2000
		total_i_class = test_loader['data'].shape[0]
		arr_index = np.arange(total_i_class)
		np.random.shuffle(arr_index)
		index = arr_index[:test_len]
		test_data = test_loader['data'][index]
		test_label = test_loader['target'][index]
		print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
		print("Confusion Matrix:", sess.run(calc_confusion, feed_dict={x: test_data, y: test_label}))


if __name__ == '__main__':
    train()