import abc
import six

from src.tf_compat import tf
from src.utils import deco_print

six.add_metaclass(abc.ABCMeta)
class ModelBase:
	"""Abstract class that defines a model. 
	"""
	def __init__(self, model_params, mode, global_step=None):
		"""Initialize a model. 

		Arguments: 
			model_params: Parameters describing a model. 
			mode: Mode. 
			global_step: Global step. 
		"""
		self._model_params = model_params
		self._mode = mode
		self._global_step = global_step if global_step is not None else tf.train.get_or_create_global_step()

	@abc.abstractmethod
	def _build_forward_pass_graph(self):
		"""Abstract method that describes how forward pass graph is constructed. 
		"""
		return

	def _build_train_op(self, loss, scope, loss_factor=1.0):
		"""Construct a training op. 

		Arguments:
			loss: Scalar 'Tensor'
		"""

		### Trainable variables
		deco_print('Trainable variables (scope=%s)' %scope)
		total_params = 0
		trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		for var in trainable_variables:
			var_params = 1
			for dim in var.get_shape():
				var_params *= dim.value
			total_params += var_params
			print('Name: {} and shape: {}'.format(var.name, var.get_shape()))
		deco_print('Number of parameters: %d' %total_params)

		### Train optimizer
		optimizer_name = self._model_params['optimizer']
		if optimizer_name == 'Momentum':
			optimizer_fn = lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.9)
		elif optimizer_name == 'AdaDelta':
			optimizer_fn = lambda lr: tf.train.AdadeltaOptimizer(lr, rho=0.95, epsilon=1e-08)
		elif optimizer_name == 'Adam':
			optimizer_fn = tf.train.AdamOptimizer
		elif optimizer_name == 'RMSProp':
			optimizer_fn = tf.train.RMSPropOptimizer
		elif optimizer_name == 'GradientDescent':
			optimizer_fn = tf.train.GradientDescentOptimizer
		else:
			raise ValueError('Unsupported optimizer: %s' % optimizer_name)

		### Learning rate decay
		if 'use_decay' in self._model_params and self._model_params['use_decay'] == True:
			learning_rate_decay_fn = lambda lr, global_step: tf.train.exponential_decay(
				learning_rate=lr,
				global_step=global_step,
				decay_steps=self._model_params['decay_steps'],
				decay_rate=self._model_params['decay_rate'],
				staircase=True)
		else:
			learning_rate_decay_fn = None

		learning_rate = self._model_params['learning_rate']
		if learning_rate_decay_fn is not None:
			learning_rate = learning_rate_decay_fn(learning_rate, self._global_step)
		optimizer = optimizer_fn(learning_rate)
		loss_scaled = loss * loss_factor
		grads_and_vars = optimizer.compute_gradients(loss_scaled, var_list=trainable_variables)

		max_grad_norm = self._model_params.get('max_grad_norm')
		if max_grad_norm is not None:
			valid_grads_and_vars = [(grad, var) for grad, var in grads_and_vars if grad is not None]
			if valid_grads_and_vars:
				grads, variables = zip(*valid_grads_and_vars)
				clipped_grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
				clipped_lookup = dict(zip(variables, clipped_grads))
				grads_and_vars = [(clipped_lookup.get(var), var) for grad, var in grads_and_vars if grad is not None]

		return optimizer.apply_gradients(grads_and_vars, global_step=self._global_step)

	@property
	def model_params(self):
		"""
		Returns:
			Parameters used to construct the model. 
		"""
		return self._model_params

	def randomInitialization(self, sess):
		sess.run(tf.global_variables_initializer())
		deco_print('Random initialization')

	def loadSavedModel(self, sess, logdir):
		if tf.train.latest_checkpoint(logdir) is not None:
			saver = tf.train.Saver(max_to_keep=100)
			saver.restore(sess, tf.train.latest_checkpoint(logdir))
			deco_print('Restored checkpoint')
		else:
			deco_print('WARNING: Checkpoint not found! Use random initialization! ')
			self.randomInitialization(sess)
