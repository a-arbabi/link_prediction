import models.tucker as tucker
import numpy as np
import tensorflow as tf
import evaluation
import data_utils

class exp_params:
  use_ncr = False
  model = 'tucker'
  consider_hypernymy_as_relation = False
  sum_decendents = False
  loss_type = 'cross entropy softmax'

def train_tucker(model, train_dataset, valid_dataset, config):
  '''
  train_sub_rel_pair_to_objs = {}
  all_sub_rel_pair_to_objs = {}
  folds = ['train', 'valid', 'test']

  for fold in folds:
    for link in datasets[fold]:
      pair = (link[0], link[1])
      if link[1] != 0:
        continue
      if pair not in all_sub_rel_pair_to_objs:
        all_sub_rel_pair_to_objs[pair] = []
      all_sub_rel_pair_to_objs[pair].append(link[2])
      
      if fold == 'train':
        if pair not in train_sub_rel_pair_to_objs:
          train_sub_rel_pair_to_objs[pair] = []
        train_sub_rel_pair_to_objs[pair].append(link[2])

  training_pairs = np.array(list(train_sub_rel_pair_to_objs.keys()))
  '''

  optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

  #dataset_size = training_pairs.shape[0] # pylint: disable=E1136  # pylint/issues/3139 


  for epoch in range(config.n_epochs):
    history = []
    #np.random.shuffle(training_pairs) #
    #for batch_start in range(0, dataset_size, config.batch_size): #
    for batch in train_dataset:
      '''
      batch = training_pairs[batch_start:batch_start + config.batch_size]
      real_batch_size = batch.shape[0]

      labels = np.zeros([real_batch_size, config.n_entities])
      for i in range(real_batch_size):
        labels[i, train_sub_rel_pair_to_objs[tuple(batch[i])]] = 1.0
      '''

      labels = batch['obj_list']

      with tf.GradientTape() as tape:
          logits = model(batch, training=True)
          #logits = model(batch, training=True) #
          #labels = tf.convert_to_tensor(labels, dtype=tf.float32) #
          n_hits = tf.reduce_sum(labels, axis=1, keepdims=True)
          labels = labels/n_hits
          if config.use_softmax:
            loss = tf.reduce_mean(
                n_hits*tf.nn.softmax_cross_entropy_with_logits(
                    labels = labels, logits = logits))
          else:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels = labels, logits = logits))

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      history.append(loss.numpy())

    print("Epoch::", epoch, ", Loss::", np.mean(history))
    print(evaluation.evaluate(model, config, valid_dataset))

def run_exp(exp_params, config, dataset_path):
  triplets, entity2idx, rel2idx = data_utils.create_datasets(dataset_path)
  gt_triplets = np.concatenate(
    [triplets[fold] for fold in ['train', 'valid', 'test']])
  config.n_entities = len(entity2idx)
  config.n_relations = len(rel2idx)

  def filter_fn(x):
    return tf.equal(x['rel'], 0)

  train_dataset = data_utils.create_aggregated_dataset(
    triplets['train'], config.n_entities)
  train_dataset = (
    train_dataset.filter(filter_fn).shuffle(500).batch(config.batch_size)).cache()

  valid_dataset = data_utils.create_aggregated_dataset(
    triplets['valid'], config.n_entities, gt_triplets)
  valid_dataset = (
    valid_dataset.filter(filter_fn).batch(config.batch_size))

  models = [tucker.TuckerModel(config) for i in range(config.n_models)]
  for model in models:
    train_tucker(model, train_dataset, valid_dataset, config)
  return models[-1]

def main():
  run_exp(exp_params, tucker.config, '../hpo_dataset')

if __name__ == '__main__':
  main()