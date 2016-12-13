import theano
import theano.tensor as T
import lasagne

import argparse
import numpy as np
import paths

# run as python LSTM.py --is_train True

def build_rnn(X):
    l_input = lasagne.layers.InputLayer(shape=X.shape)

    #not sure if we need dropout
    l_dropout_input = lasagne.layers.DropoutLayer(l_input, p=0.5)

    l_lstm = lasagne.layers.LSTMLayer(l_dropout_input, 100, unroll_scan=True,
                                  grad_clipping=5.)
    l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)
    l_output = lasagne.layers.DenseLayer(l_dropout_output, nonlinearity=lasagne.nonlinearities.softmax)

    #Discuss structure
    # l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size))
    # l_forward_1 = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,nonlinearity=lasagne.nonlinearities.tanh)
    # l_forward_2 = lasagne.layers.LSTMLayer(l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP, nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
    #l_out = lasagne.layers.DenseLayer(l_forward_2, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)


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
        #y_train, X_val, y_val, X_test, y_test = load_dataset()
    else:
        print("----------------------------- Loading validation data -----------------------------")
        train = np.load(paths.PROCESSED_FOLDER + 'merged_val.npy')
        x = train[:, 0]
        y = train[:, 1]

    print('Placeholder images')
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print('Build rnn')
    # Create model
    network = build_rnn(x)

    #Loss
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    #updates
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
    #this or adagrad???
    #updates = lasagne.updates.adagrad(loss, params, LEARNING_RATE)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


    print("Starting training...")

    # Iterate over epochs:
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
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
