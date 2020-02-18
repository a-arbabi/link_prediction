import models.tucker as tucker
import numpy as np
import tensorflow as tf
import evaluation
import data_utils
import sys
import os

class exp_params:
  use_ncr = False
  model = 'tucker'
  consider_hypernymy_as_relation = False
  sum_decendents = False
  loss_type = 'cross entropy softmax'

def print_results(results, output_file=None):
  if output_file is None:
    output_file = sys.stdout
  for cat in results:
    print(cat, ':', results[cat].numpy(), end='\t', file=output_file)
  print('', file=output_file)

def train_tucker(model, train_dataset, valid_dataset, config):
  @tf.function
  def apply_batch(model, batch, config, optimizer):
    labels = batch['obj_list']

    with tf.GradientTape() as tape:
        logits = model(batch, training=True)
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
    return loss

  optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

  best_results = {}
  best_weights = None
  for epoch in range(config.n_epochs):
    history = []
    for batch in train_dataset:
      loss = apply_batch(model, batch, config, optimizer)
      history.append(loss.numpy())

    print("Epoch::", epoch, ", Loss::", np.mean(history))
    if (epoch+1)%5==0:
      valid_results = evaluation.evaluate(model, config, valid_dataset)
      if len(best_results) == 0 or best_results['MRR'] < valid_results['MRR']:
        best_results = valid_results
        best_weights = model.get_weights()
      print_results(valid_results)
  model.set_weights(best_weights)
  return best_results
      
def run_exp(exp_params, config, dataset_path):
  triplets, entity2idx, rel2idx, sp_mat = data_utils.create_datasets(dataset_path)
  del(sp_mat)
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

  '''
  models = [tucker.TuckerModel(config) for i in range(config.n_models)]
  for model in models:
    train_tucker(model, train_dataset, valid_dataset, config)
  '''

  model = tucker.TuckerModel(config)
  best_result = train_tucker(model, train_dataset, valid_dataset, config)
  print_results(best_result)
  return model, best_result
#  return models[-1]

def grid_search_exp(exp_params, dataset_path, output_path):
  config = tucker.config

  config.use_core_tensor = False
  with open(output_path, 'w') as output_file:
    #for config.use_core_tensor in [True]:
    for config.lr in [0.001, 0.002]:
      for config.batch_size in [128, 256]:
        for config.d_entity in [100, 250, 500]:
          model, best_result = run_exp(exp_params, config, dataset_path)
          del(model)
#          output_file.write('use_core_tensor: ' + str(config.use_core_tensor))
          output_file.write('\tlr: ' + str(config.lr))
          output_file.write('\tbatch_size: ' + str(config.batch_size))
          output_file.write('\td_entity: ' + str(config.d_entity))
          output_file.write('\t')
          print_results(best_result, output_file)
          output_file.flush()


def main():
  grid_search_exp(exp_params, '../hpo_dataset', '../grid_search_result_no_core_tensor.txt')
  #run_exp(exp_params, tucker.config, '../hpo_dataset')

if __name__ == '__main__':
  main()