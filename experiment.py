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

def train_tucker(model, train_dataset, valid_dataset, config):
  optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

  for epoch in range(config.n_epochs):
    history = []
    for batch in train_dataset:
      loss = apply_batch(model, batch, config, optimizer)
      history.append(loss.numpy())

    print("Epoch::", epoch, ", Loss::", np.mean(history))
    valid_results = evaluation.evaluate(model, config, valid_dataset)
    for cat in valid_results:
      print(cat, valid_results[cat].numpy())
    print('')

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

  models = [tucker.TuckerModel(config) for i in range(config.n_models)]
  for model in models:
    train_tucker(model, train_dataset, valid_dataset, config)
  return models[-1]

def main():
  run_exp(exp_params, tucker.config, '../hpo_dataset')

if __name__ == '__main__':
  main()