import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

SETTING_DEFAULT = 'ff'
PLOT_TYPE_DEFAULT = 'all'

def plot_loss():
    # Read data from CSV
    train_data = np.genfromtxt('./'+FLAGS.setting+'/results.csv', delimiter=',', usecols=(0,1), names=True)
    test_data = np.genfromtxt('./'+FLAGS.setting+'/test_results.csv', delimiter=',', usecols=(0,1), names=True)
    val_data = np.genfromtxt('./'+FLAGS.setting+'/validation_results.csv', delimiter=',', usecols=(0,1), names=True)

    # Create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)

    # Plot data
    ax.plot(train_data['Step'], train_data['Loss'], color='g', label='train')
    ax.plot(test_data['Step'], test_data['Loss'], color='r', label='test')
    ax.plot(val_data['Step'], val_data['Loss'], color='b', label='validation')

    # Plot details
    plt.title('Cross entropy loss')
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.legend(loc='upper right')

    # Save
    plt.savefig('./'+FLAGS.setting+'/loss.png', format='png')

def plot_reg_loss():
    # Read data from CSV
    train_data = np.genfromtxt('./'+FLAGS.setting+'/results.csv', delimiter=',', usecols=(0,2), names=True)
    print(train_data.dtype.names)

    # Create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)

    # Plot data
    ax.plot(train_data['Step'], train_data['Gradient_Norm'], color='g', label='train')

    # Plot details
    plt.title('L2 regularized loss')
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.legend(loc='upper right')

    # Save
    plt.savefig('./'+FLAGS.setting+'/regularized_loss.png', format='png')

def plot_bleu_1():
    # Read data from CSV
    test_data = np.genfromtxt('./'+FLAGS.setting+'/test_results.csv', delimiter=',', usecols=(0,2), names=True)
    val_data = np.genfromtxt('./'+FLAGS.setting+'/validation_results.csv', delimiter=',', usecols=(0,2), names=True)

    # Create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)

    # Plot data
    ax.plot(test_data['Step'], test_data['Bleu1'], color='r', label='test')
    ax.plot(val_data['Step'], val_data['Bleu1'], color='b', label='validation')

    # Plot details
    plt.title('Bleu-1 score')
    plt.ylabel('score')
    plt.xlabel('step')
    plt.legend(loc='lower right')

    # Save
    plt.savefig('./'+FLAGS.setting+'/Bleu1.png', format='png')


def plot_bleu_4():
    # Read data from CSV
    test_data = np.genfromtxt('./'+FLAGS.setting+'/test_results.csv', delimiter=',', usecols=(0,3), names=True)
    val_data = np.genfromtxt('./'+FLAGS.setting+'/validation_results.csv', delimiter=',', usecols=(0,3), names=True)

    # Create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)

    # Plot data
    ax.plot(test_data['Step'], test_data['Bleu4'], color='r', label='test')
    ax.plot(val_data['Step'], val_data['Bleu4'], color='b', label='validation')

    # Plot details
    plt.title('Bleu-4 score')
    plt.ylabel('score')
    plt.xlabel('step')
    plt.legend(loc='lower right')

    # Save
    plt.savefig('./'+FLAGS.setting+'/Bleu4.png', format='png')


def plot_bleu():
    # Read data from CSV
    test_data = np.genfromtxt('./'+FLAGS.setting+'/test_results.csv', delimiter=',', usecols=(0,2,3), names=True)
    val_data = np.genfromtxt('./'+FLAGS.setting+'/validation_results.csv', delimiter=',', usecols=(0,2,3), names=True)

    # Create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)

    # Plot data
    ax.plot(test_data['Step'], test_data['Bleu1'], color='r', label='BLEU-1 test')
    ax.plot(val_data['Step'], val_data['Bleu1'], color='b', label='BLEU-1 validation')
    ax.plot(test_data['Step'], test_data['Bleu4'], color='r', linestyle='dashed', label='BLEU-4 test')
    ax.plot(val_data['Step'], val_data['Bleu4'], color='b', linestyle='dashed', label='BLEU-4 validation')

    # Plot details
    plt.title('Bleu-1 and Bleu-4 scores')
    plt.ylabel('score')
    plt.xlabel('step')
    plt.legend(loc='center right')

    # Save
    plt.savefig('./'+FLAGS.setting+'/Bleu.png', format='png')


def main(_):
    if FLAGS.plot_type == 'loss':
        plot_loss()
    elif FLAGS.plot_type == 'bleu-1':
        plot_bleu_1()
    elif FLAGS.plot_type == 'bleu-4':
        plot_bleu_4()
    elif FLAGS.plot_type == 'all':
        plot_loss()
        plot_reg_loss()
        plot_bleu_1()
        plot_bleu_4()
        plot_bleu()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--setting', type = str, default = SETTING_DEFAULT,
                      help='Setting to get the results from')
    parser.add_argument('--plot_type', type = str, default = PLOT_TYPE_DEFAULT,
                      help='Type of plot. Possible options: loss or bleu-1 or bleu-4 or all')
    FLAGS, unparsed = parser.parse_known_args()

    main(None)
