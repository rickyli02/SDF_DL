import tensorflow as _tf

_tf.compat.v1.disable_v2_behavior()

tf = _tf.compat.v1
Dense = _tf.compat.v1.keras.layers.Dense
BasicRNNCell = _tf.compat.v1.nn.rnn_cell.BasicRNNCell
GRUCell = _tf.compat.v1.nn.rnn_cell.GRUCell
LSTMCell = _tf.compat.v1.nn.rnn_cell.LSTMCell
DropoutWrapper = _tf.compat.v1.nn.rnn_cell.DropoutWrapper
MultiRNNCell = _tf.compat.v1.nn.rnn_cell.MultiRNNCell
LSTMStateTuple = _tf.compat.v1.nn.rnn_cell.LSTMStateTuple
