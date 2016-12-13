import theano
import theano.tensor as T
import lasagne

import argparse
import numpy as np
import paths
import time

FEATURES_DIMENSION = 4096
CAPTION_DIMENSION = 20 # arbitrary for now
NUM_UNITS = 10 # arbitrary for now

# run as python LSTM.py --is_train True

def build_rnn():
    l_input = lasagne.layers.InputLayer(shape=(None, FEATURES_DIMENSION))

    l_dropout_input = lasagne.layers.DropoutLayer(l_input, p=0.5)
    l_lstm = lasagne.layers.LSTMLayer(l_dropout_input, NUM_UNITS, unroll_scan=False, grad_clipping=5.0)
    l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)
    l_output = lasagne.layers.DenseLayer(l_dropout_output, CAPTION_DIMENSION, nonlinearity=lasagne.nonlinearities.softmax)

    return l_output

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(_):
    # Load the dataset
    if FLAGS.is_train == 'True':
        print("----------------------------- Loading training data -----------------------------")
        train = np.load(paths.PROCESSED_FOLDER + 'merged_train.npy')
        x = train[:, 0]
        y = train[:, 1]
    else:
        print("----------------------------- Loading validation data -----------------------------")
        train = np.load(paths.PROCESSED_FOLDER + 'merged_val.npy')
        x = train[:, 0]
        y = train[:, 1]

    print('Placeholder variables')
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print('Build rnn')
    # Input layer
    l_input = lasagne.layers.InputLayer(shape=(None, None, FEATURES_DIMENSION))
    # Get shape
    batch_size, seq_length, _ = l_input.input_var.shape
    # RNN layer
    l_lstm = lasagne.layers.LSTMLayer(l_input, num_units=NUM_UNITS)
    # Flatten
    l_shp = lasagne.layers.ReshapeLayer(l_lstm, (-1, NUM_UNITS))
    # Fully connected layer
    l_dense = lasagne.layers.DenseLayer(l_shp, num_units=CAPTION_DIMENSION, nonlinearity=lasagne.nonlinearities.softmax)
    # Get original shape back
    l_output = lasagne.layers.ReshapeLayer(l_dense, (batch_size, seq_length, CAPTION_DIMENSION))

    print('Build prediction and loss')
    #Loss
    output = lasagne.layers.get_output(l_output)
    prediction = output[:, -1]
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    print('Build optimizer')
    #updates
    params = lasagne.layers.get_all_params(l_output, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
    #updates = lasagne.updates.adagrad(loss, params, LEARNING_RATE)

    print('Build prediction and loss for testing')
    test_output = lasagne.layers.get_output(l_output, deterministic=True)
    test_prediction = test_output[:, -1]
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    print('Build accuracy for testing')
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    print('----------------------------- Training -----------------------------')
    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([l_input.input_var, target_var], loss, updates=updates)

    print('hiiiiii, jeijei')

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([l_input.input_var, target_var], loss)

    print("Starting training...")

    # Iterate over epochs:
    for epoch in range(100):
        print("Epoch ", epoch)
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x, y, 500, shuffle=True):
            print("Iterating batch ...")
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--is_train', type = str, default = True,
                      help='Training or Testing')

  FLAGS, unparsed = parser.parse_known_args()


  print_flags()
  main(None)
