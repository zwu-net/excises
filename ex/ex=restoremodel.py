
import tensorflow as tf
sess = tf.Session()
new_saver = tf.train.import_meta_graph('/temp/test.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('/temp'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_)