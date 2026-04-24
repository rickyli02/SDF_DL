import os
import json
import numpy as np

from src.data import data_layer
from src.model.model_GAN import FeedForwardModelWithNA_GAN
from src.tf_compat import tf
from src.utils import deco_print

tf.flags.DEFINE_string('config', '', 'Path to the file with configurations')
tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')

tf.flags.DEFINE_integer('saveBestFreq', -1, 'Frequency to save best model')
tf.flags.DEFINE_boolean('printOnConsole', True, 'Print on console or not')
tf.flags.DEFINE_boolean('saveLog', True, 'Save log or not')
tf.flags.DEFINE_integer('printFreq', 128, 'Frequency to print on console')
tf.flags.DEFINE_integer('ignoreEpoch', 64, 'Ignore first several epochs')

FLAGS = tf.flags.FLAGS


def validate_config_paths(config):
	required_paths = [
		config['individual_feature_file'],
		config['individual_feature_file_valid'],
		config['individual_feature_file_test'],
		config['macro_feature_file'],
		config['macro_feature_file_valid'],
		config['macro_feature_file_test'],
	]
	missing = [path for path in required_paths if not os.path.exists(path)]
	if missing:
		raise FileNotFoundError(
			'Missing dataset files:\n%s\n\nDownload the dataset bundle linked in README.md and place it under the repo so these relative paths resolve.'
			% '\n'.join(missing)
		)

def main(_):
	with open(FLAGS.config, 'r') as file:
		config = json.load(file)
		if not 'macro_idx' in config:
			config['macro_idx'] = None
	validate_config_paths(config)
	deco_print('Read the following in config: ')
	print(json.dumps(config, indent=4))

	deco_print('Creating data layer')
	dl = data_layer.DataInRamInputLayer(
		config['individual_feature_file'],
		pathMacroFeature=config['macro_feature_file'],
		macroIdx=config['macro_idx'])
	meanMacroFeature, stdMacroFeature = dl.getMacroFeatureMeanStd()
	dl_valid = data_layer.DataInRamInputLayer(
		config['individual_feature_file_valid'],
		pathMacroFeature=config['macro_feature_file_valid'],
		macroIdx=config['macro_idx'], 
		meanMacroFeature=meanMacroFeature,
		stdMacroFeature=stdMacroFeature)
	dl_test = data_layer.DataInRamInputLayer(
		config['individual_feature_file_test'],
		pathMacroFeature=config['macro_feature_file_test'],
		macroIdx=config['macro_idx'], 
		meanMacroFeature=meanMacroFeature,
		stdMacroFeature=stdMacroFeature)
	if config['weighted_loss']:
		loss_weight = dl.getDateCountList()
		loss_weight_valid = dl_valid.getDateCountList()
		loss_weight_test = dl_test.getDateCountList()
	else:
		loss_weight = None
		loss_weight_valid = None
		loss_weight_test = None
	deco_print('Data layer created')

	global_step = tf.train.get_or_create_global_step()
	model = FeedForwardModelWithNA_GAN(config, 'train', config['tSize'], global_step=global_step)
	model_valid = FeedForwardModelWithNA_GAN(config, 'valid', config['tSize_valid'], force_var_reuse=True, global_step=global_step)
	model_test = FeedForwardModelWithNA_GAN(config, 'test', config['tSize_test'], force_var_reuse=True, global_step=global_step)
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess_config = tf.ConfigProto(gpu_options=gpu_options)
	sess = tf.Session(config=sess_config)
	model.randomInitialization(sess)
	
	sharpe_train, sharpe_valid, sharpe_test = model.train(sess, dl, dl_valid, FLAGS.logdir, model_valid, 
		loss_weight=loss_weight, loss_weight_valid=loss_weight_valid,
		dl_test=dl_test, model_test=model_test, loss_weight_test=loss_weight_test, 
		printOnConsole=FLAGS.printOnConsole, printFreq=FLAGS.printFreq, saveLog=FLAGS.saveLog, 
		saveBestFreq=FLAGS.saveBestFreq, ignoreEpoch=FLAGS.ignoreEpoch)

	### best model on sharpe
	idxBestEpoch = np.array(sharpe_valid).argmax()
	sharpe_train_best_sharpe = sharpe_train[idxBestEpoch]
	sharpe_valid_best_sharpe = sharpe_valid[idxBestEpoch]
	sharpe_test_best_sharpe = sharpe_test[idxBestEpoch]
	deco_print('SDF Portfolio Sharpe Ratio (Evaluated on Sharpe): Train %0.3f\tValid %0.3f\tTest %0.3f' %(sharpe_train_best_sharpe, sharpe_valid_best_sharpe, sharpe_test_best_sharpe))

if __name__ == '__main__':
	tf.app.run()
