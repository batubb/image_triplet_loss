import tensorflow as tf
from preprocessing import PreProcessing
from model import TripletLoss
import numpy as np



def computeCosSimilarity(output1, output2):
    dot_product = tf.reduce_sum(output1*output2, 1)
    norm_1 = tf.sqrt(tf.reduce_sum(tf.pow(output1, 2), 1))
    norm_2 = tf.sqrt(tf.reduce_sum(tf.pow(output2, 2), 1))
    return dot_product / (norm_1 * norm_2)


def computeAccuracyGreat(thresholdTo1, v):
    hit = 0
    total = 0
    for elem in v:
        if elem > thresholdTo1:
            hit += 1
        total += 1
    return hit*1.0 / total 

def computeAccuracyLess(thresholdTo1, v):
    hit = 0
    total = 0
    for elem in v:
        if elem < thresholdTo1:
            hit += 1
        total += 1
    return hit*1.0 / total 



'''
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('train_iter', 250, 'Total training iter')
flags.DEFINE_integer('step', 50, 'Save after ... iteration')
flags.DEFINE_float('learning_rate','0.01','Learning rate')
flags.DEFINE_float('momentum','0.99', 'Momentum')
flags.DEFINE_string('model', 'conv_net', 'model to run')
flags.DEFINE_string('data_src', './ChallengeImages2/', 'source of training dataset')
'''



def train():
    data_src = './ChallengeImages2/'
    learning_rate = 0.01
    train_iter = 250
    batch_size = 128
    momentum = 0.99
    step = 50
    # Setup Dataset
    dataset = PreProcessing(data_src)
    print(dataset.images_test)
    print(dataset.labels_test)
    model = TripletLoss()
    height = 50
    width = 150
    dims = [height, width, 3]
    placeholder_shape = [None] + dims
    print("placeholder_shape", placeholder_shape)

    # Setup Network
    next_batch = dataset.get_triplets_batch
    anchor_input = tf.placeholder(tf.float32, placeholder_shape, name='anchor_input')
    positive_input = tf.placeholder(tf.float32, placeholder_shape, name='positive_input')
    negative_input = tf.placeholder(tf.float32, placeholder_shape, name='negative_input')

    margin = 0.5
    # Will be of size N x 28
    anchor_output = model.conv_net(anchor_input, reuse=tf.AUTO_REUSE)
    positive_output = model.conv_net(positive_input, reuse=tf.AUTO_REUSE)
    negative_output = model.conv_net(negative_input, reuse=tf.AUTO_REUSE)
    print(tf.shape(anchor_output))
    return

    '''
    compute the similarity between the positive_output and the negative_output
    if similarity is < 0.25 that's great
    '''
    similarity_pos_neg = computeCosSimilarity(positive_output, negative_output)
    similarity_pos_anchor = computeCosSimilarity(positive_output, anchor_output)

    loss = model.triplet_loss(anchor_output, positive_output, negative_output, margin)

    # Setup Optimizer
    global_step = tf.Variable(0, trainable=False)

    train_step = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(loss,
                                                                                                             global_step=global_step)

    # Start Training
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Setup Tensorboard
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train.log', sess.graph)

        # Train iter
        for i in range(train_iter):
            batch_anchor, batch_positive, batch_negative = next_batch(batch_size)

            _, l, summary_str = sess.run([train_step, loss, merged],
                                         feed_dict={anchor_input: batch_anchor, positive_input: batch_positive, negative_input: batch_negative})

            #pNAccuracy = computeAccuracyLess(0.25, similarity_p_n)
            #pAAccuracy = computeAccuracyGreat(0.75, similarity_p_a)
            writer.add_summary(summary_str, i)
            print("#%d - Loss" % i, l)
            #print("#%d - P and A Accuracy" % i, pAAccuracy)
            #print("#%d - N and A Accuracy" % i, pNAccuracy)

            if (i + 1) % step == 0:
                saver.save(sess, "model_triplet/model.ckpt")
        saver.save(sess, "model_triplet/model.ckpt")
    print('Training completed successfully.')
    return dataset


if __name__ == "__main__":
    train()