# Logistic regression example in TF using Kaggle's Titanic Dataset.
# Download tfformbook-logistic-train.csv from https://www.kaggle.com/c/titanic/data

import tensorflow as tf
import os

# same params and variables initialization as log reg.
W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")


# former inference is now used for combining inputs
def combine_inputs(X):
    return tf.matmul(X, W) + b


# new inferred value is the sigmoid applied to the former
def inference(X):
    return tf.sigmoid(combine_inputs(X))


def loss(X, Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)



def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(3, "../data/tfformbook-logistic-train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])

    return  name, survived


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):

    predicted = tf.cast(inference(X) > 0.5, tf.float32)

    print (sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))

# Launch the graph in a session, setup boilerplate
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    X, Y = inputs()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(X.eval())

    Z = tf.reshape(X, [3, 1])
    print(Z.eval())
    coord.request_stop()
    coord.join(threads)
    sess.close()
