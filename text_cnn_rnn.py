import numpy as np
import tensorflow as tf

class TextCNNRNN(object):
	def __init__(self, embedding_mat, non_static, hidden_unit, sequence_length, max_pool_size,
		num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):


		self.input_x = tf.compat.v1.placeholder(tf.int32, [None, sequence_length], name='input_x')
		self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name='input_y')
		self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name='dropout_keep_prob')
		self.batch_size = tf.compat.v1.placeholder(tf.int32, [])
		self.pad = tf.compat.v1.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
		self.real_len = tf.compat.v1.placeholder(tf.int32, [None], name='real_len')

		l2_loss = tf.constant(0.0)

		with tf.device('/cpu:0'), tf.compat.v1.name_scope('embedding'):
			if not non_static:
				W = tf.constant(embedding_mat, name='W')
			else:
				W = tf.Variable(embedding_mat, name='W')
			self.embedded_chars = tf.nn.embedding_lookup(params=W, ids=self.input_x)
			emb = tf.expand_dims(self.embedded_chars, -1)

		pooled_concat = []
		reduced = np.int32(np.ceil((sequence_length) * 1.0 / max_pool_size))
		
		for i, filter_size in enumerate(filter_sizes):
			with tf.compat.v1.name_scope('conv-maxpool-%s' % filter_size):

				# Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
				num_prio = (filter_size-1) // 2
				num_post = (filter_size-1) - num_prio
				pad_prio = tf.concat([self.pad] * num_prio,1)
				pad_post = tf.concat([self.pad] * num_post,1)
				emb_pad = tf.concat([pad_prio, emb, pad_post],1)

				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name='W')
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
				conv = tf.nn.conv2d(input=emb_pad, filters=W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

				h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

				# Maxpooling over the outputs
				pooled = tf.nn.max_pool2d(input=h, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1], padding='SAME', name='pool')
				pooled = tf.reshape(pooled, [-1, reduced, num_filters])
				pooled_concat.append(pooled)

		pooled_concat = tf.concat(pooled_concat,2)
		pooled_concat = tf.nn.dropout(pooled_concat, 1 - (self.dropout_keep_prob))


		lstm_cell =tf.keras.layers.LSTMCell(units=hidden_unit)
		lstm_cell = tf.nn.RNNCellDropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)

		self._initial_state = lstm_cell.get_initial_state(self.batch_size, tf.float32)
		
		self._initial_state = get_initial_state(lstm_cell,self.batch_size, tf.float32)
		inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(pooled_concat,num_or_size_splits=int(reduced),axis=1)]
		return_state =tf.keras.layers.RNN(lstm_cell, inputs, initial_state=self._initial_state,return_sequences=True, return_state=True)

		#print(return_state)
		# Collect the appropriate last words into variable output (dimension = batch x embedding_size)
		output = return_state[0]
		with tf.compat.v1.variable_scope('Output'):
			tf.compat.v1.get_variable_scope().reuse_variables()
			one = tf.ones([1, hidden_unit], tf.float32)
			for i in range(1,len(outputs)):
				ind = self.real_len < (i+1)
				ind = tf.cast(ind, dtype=tf.float32) 
				ind = tf.expand_dims(ind, -1)
				mat = tf.matmul(ind, one)
				output = tf.add(tf.multiply(output, mat),tf.multiply(outputs[i], 1.0 - mat))

		with tf.compat.v1.name_scope('output'):
			self.W = tf.Variable(tf.random.truncated_normal([hidden_unit, num_classes], stddev=0.1), name='W')
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.compat.v1.nn.xw_plus_b(output, self.W, b, name='scores')
			self.predictions = tf.argmax(input=self.scores, axis=1, name='predictions')

		with tf.compat.v1.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = tf.stop_gradient( self.input_y), logits = self.scores) #  only named arguments accepted            
			self.loss = tf.reduce_mean(input_tensor=losses) + l2_reg_lambda * l2_loss

		with tf.compat.v1.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(input=self.input_y, axis=1))
			self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_predictions, "float"), name='accuracy')

		with tf.compat.v1.name_scope('num_correct'):
			correct = tf.equal(self.predictions, tf.argmax(input=self.input_y, axis=1))
			self.num_correct = tf.reduce_sum(input_tensor=tf.cast(correct, 'float'))

