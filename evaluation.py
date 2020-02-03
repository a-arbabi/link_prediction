import numpy as np
import tensorflow as tf

def evaluate(model, config, dataset):
  ranks = []
  for batch in dataset:
    logits = model(batch, training=False) 
    probs = tf.nn.softmax(logits, axis=1)
    negatives = 1.0 - batch['gt_obj_list']
    negative_probs = probs * negatives
    indecies = tf.where(batch['obj_list'])
    gathered_probs = tf.gather_nd(probs, indecies)
    gathered_negs = tf.gather(negative_probs, indecies[:,0])
    batch_ranks = tf.math.count_nonzero(
      tf.expand_dims(gathered_probs, -1) <= gathered_negs, axis=1)+1
    ranks.append(batch_ranks)

  ranks = tf.concat(ranks, 0)
  return {
      'R@1': tf.reduce_mean(tf.cast(tf.equal(ranks, 1), tf.float32)),
      'R@3': tf.reduce_mean(tf.cast(tf.less_equal(ranks, 3), tf.float32)),
      'R@10': tf.reduce_mean(tf.cast(tf.less_equal(ranks, 10), tf.float32)),
      'MRR': tf.reduce_mean(1.0/tf.cast(ranks, tf.float32)),
  }