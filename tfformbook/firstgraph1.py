import tensorflow as tf

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.mul(a,b, name="mul_c")
d = tf.add(a,b, name="add_d")
e = tf.add(c,d, name="add_e")

with tf.Session() as sess:
    sess.run(e)
    sess.run(a)
    n = a.eval()

    print(type(n))

    print (a.eval())
    #n = tf.Print(a, [a], "message this is")
    #print (n)
    print (e.eval())