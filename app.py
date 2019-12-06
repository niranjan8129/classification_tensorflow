import os
import sys
import json
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN
from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
from werkzeug import secure_filename

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv'])

logging.getLogger().setLevel(logging.INFO)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Bootstrap(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	#trained_dir = sys.argv[1]
	trained_dir = './trained_results_1575177514/' 
	
	#if not trained_dir.endswith('/'):
	#	trained_dir += '/'
	#test_file = sys.argv[2]
	
	if request.method == 'POST':
		file = request.files['ReceivedFile']
		logging.critical('Received Filename from App: {}'.format(file))
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			test_file = './uploads/'+filename
		
		
			df = pd.read_csv(test_file)
			text = df[['Descript']]
				   
			params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
			flag = 0
			x_, y_, df = load_test_data(test_file, labels,flag)
			x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
			x_ = map_word_to_index(x_, words_index)
		else:
			#text = request.json["text"]
			text =request.form["query"]
			params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
			flag = 1
			x_, y_, df = load_test_data(text, labels,flag)
			x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
			x_ = map_word_to_index(x_, words_index)

	x_test, y_test = np.asarray(x_), None
	if y_ is not None:
		y_test = np.asarray(y_)

	timestamp = trained_dir.split('/')[-2].split('_')[-1]
	predicted_dir = './predicted_results_' + timestamp + '/'
	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
	os.makedirs(predicted_dir)

	with tf.Graph().as_default():
		session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.compat.v1.Session(config=session_conf)
		with sess.as_default():
			cnn_rnn = TextCNNRNN(
				embedding_mat = embedding_mat,
				non_static = params['non_static'],
				hidden_unit = params['hidden_unit'],
				sequence_length = len(x_test[0]),
				max_pool_size = params['max_pool_size'],
				filter_sizes = map(int, params['filter_sizes'].split(",")),
				num_filters = params['num_filters'],
				num_classes = len(labels),
				embedding_size = params['embedding_dim'],
				l2_reg_lambda = params['l2_reg_lambda'])

			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

			def predict_step(x_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.dropout_keep_prob: 1.0,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				predictions = sess.run([cnn_rnn.predictions], feed_dict)
				return predictions

			checkpoint_file = trained_dir + 'best_model.ckpt'
			saver = tf.compat.v1.train.Saver(tf.compat.v1.all_variables())
			saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			logging.critical('{} has been loaded'.format(checkpoint_file))

			batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)

			predictions, predict_labels = [], []
			for x_batch in batches:
				batch_predictions = predict_step(x_batch)[0]
				for batch_prediction in batch_predictions:
					predictions.append(batch_prediction)
					predict_labels.append(labels[batch_prediction])
					logging.critical('Prediction is complete Class belongs to: {}'.format(predict_labels[0]))

			#if os.path.exists(test_file):
			#	os.remove(test_file)
				
			return render_template('results.html',prediction = predict_labels[0],name =text)

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)
	return params, words_index, labels, embedding_mat

def load_test_data(test_file, labels,flag):
	#df = pd.read_csv(test_file, sep='|')
	logging.critical('{} test_file received '.format(test_file))
	select = ['Descript']
	if flag == 0:
		df = pd.read_csv(test_file)
		df = df.dropna(axis=0, how='any', subset=select)
		test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()
	else: 
		df = pd.Series(test_file)
		df =pd.DataFrame(df.values, columns=[select]) 
		test_examples = df.iloc[0].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

	logging.critical('{} df received '.format(df))

	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_ = None
	if 'Category' in df.columns:
		select.append('Category')
		y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

	not_select = list(set(df.columns) - set(select))
	df = df.drop(not_select, axis=1)
	return test_examples, y_, df

def map_word_to_index(examples, words_index):
	x_ = []
	for example in examples:
		temp = []
		for word in example:
			if word in words_index:
				temp.append(words_index[word])
			else:
				temp.append(0)
		x_.append(temp)
	return x_

if __name__ == '__main__':
	app.run(debug=True)