import tensorflow as tf
import numpy as np

p = tf.constant([[1, 2, 3],
				[4, 5, 6],
				], dtype=tf.float32)
n = tf.constant([[11, 12, 13],
				[-21, 47, 1],
				], dtype=tf.float32)
int1 = p*n
dot_product=tf.reduce_sum(p*n, 1)
int2 = tf.sqrt(tf.reduce_sum(tf.pow(p, 2), 1))
int3 = tf.sqrt(tf.reduce_sum(tf.pow(n, 2), 1))
int4 = int2*int3
similarity = dot_product/int4




with tf.Session() as sess:
	res1, res2, res3, res4, res5, res6 = sess.run([int1, dot_product, int2, int3, int4, similarity])
	print(res1)
	print(res2)
	print(res3)
	print(res4)
	print(res5)
	print(res6)
