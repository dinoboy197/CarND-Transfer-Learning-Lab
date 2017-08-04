import pickle
import tensorflow as tf
# import Keras layers you need here
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    import numpy as np
    
    num_classes = len(np.unique(y_train))

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    # train your model here
    input_shape = X_train.shape[1:]
    #inp = Input(shape=input_shape)
    
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    #model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.5))
    #model.add(Activation('relu'))
    #model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # preprocess data
    #X_normalized = np.array(X_train / 255.0 - 0.5 )
    
    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)
    
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_train, y_one_hot, nb_epoch=FLAGS.epochs, validation_split=0.2)
    
    # preprocess data
    #X_normalized_test = np.array(X_test / 255.0 - 0.5 )
    y_one_hot_val = label_binarizer.fit_transform(y_val)
    
    print("Testing")
    
    # Evaluate the test data in Keras Here
    metrics = model.evaluate(X_val, y_one_hot_val)
    # UNCOMMENT CODE
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
