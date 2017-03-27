# Linear regression example in TF.

import tensorflow as tf


W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")


def inference(X):
    return tf.matmul(X, W) + b



def loss(X, Y):
    Y_predicted = inference(X)
    #return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))

    return tf.sqrt(tf.reduce_mean((Y-Y_predicted)**2))


def inputs():
    # Data from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]


    file = open("/tmp/test.txt", "w")
    for i in range(len(weight_age)):
        file.write(str(blood_fat_content[i])+" ")
        file.write("1:"+ str(weight_age[i][0]) +" ")
        file.write("2:" + str(weight_age[i][1])+"\n")
    file.close()


    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


def train(total_loss):
    learning_rate = 0.01
    #learning_rate=.5
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    t1 = [[80., 25.]];
    print (t1, sess.run(inference(t1))) # ~ 303
    print (sess.run(inference([[65., 25.]]))) # ~ 256

# Launch the graph in a session, setup boilerplate
X, Y = inputs()

total_loss = loss(X, Y)
train_op = train(total_loss)
model = tf.global_variables_initializer()
with tf.Session() as sess:
   sess.run(model)
   #print(X)
   training_steps = 10000
   for step in range(training_steps):
       sess.run([train_op])
       if step % 10 == 0:
           print("loss: ", sess.run([total_loss]))

   print("w=", W.eval(), "b=", b.eval())



