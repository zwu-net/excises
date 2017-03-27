
import os
import tensorflow as tf

filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), "tfformbook-logistic-train.csv")])

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)
print(key)
# decode_csv will convert a Tensor from type string (the text line) in
# a tuple of tensor columns with the specified defaults, which also
# sets the data type for each column
record_defaults=[[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]]

decoded = tf.decode_csv(value, record_defaults=record_defaults)

passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = tf.train.batch(decoded,
                                  3,
                                 )

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(3):
        fea1, fea2, key1, value1 = sess.run([passenger_id, survived, key, value])
        print("original", str(fea1) + " " + str(fea2))
        print("reshape fea1", (tf.reshape(fea1, [3,1])).eval())
        print("transpose", tf.transpose(fea1, [0]).eval())
        print(key1)
        print(value1)
        print (tf.stack([fea1, fea2]).eval())
        features = tf.transpose(tf.stack([fea1, fea2]))
        print (features.eval())

    coord.request_stop()
    coord.join(threads)

