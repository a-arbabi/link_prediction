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

'''
def evaluate(model, config, all_sub_rel_pair_to_objs, dataset):
  ranks = []
  dataset_size = dataset.shape[0]
  
  for batch_start in range(0, dataset_size, config.batch_size):
    batch = dataset[batch_start:batch_start + config.batch_size]
    real_batch_size = batch.shape[0]
    
    labels = np.ones([real_batch_size, config.n_entities])
    for i in range(real_batch_size):
      pair = (batch[i,0], batch[i,1])
      if pair[1]!=0:
        continue
      labels[i, all_sub_rel_pair_to_objs[pair]] = 0.0
    
    logits = model(batch[:,:2], training=False).numpy()
      
    predicted_logit = logits[np.arange(real_batch_size), batch[:,2]]
    predicted_logit = np.expand_dims(predicted_logit, 1)
    compares = (logits>= predicted_logit)
    ranks.append(np.sum(compares*labels, 1) + 1) 
      
  ranks = np.concatenate(ranks)
  
  return {
      'R@1': np.mean(ranks==1),
      'R@3': np.mean(ranks<=3),
      'R@10': np.mean(ranks<=10),
      'MRR': np.mean(1.0/ranks)
  }

'''