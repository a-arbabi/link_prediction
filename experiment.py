import models.tucker as tucker
import numpy as np
import tensorflow as tf
import evaluation

def train_tucker(model, datasets, config):

  train_sub_rel_pair_to_objs = {}
  all_sub_rel_pair_to_objs = {}
  folds = ['train', 'valid', 'test']

  for fold in folds:
    for link in datasets[fold]:
      pair = (link[0], link[1])
      if pair not in all_sub_rel_pair_to_objs:
        all_sub_rel_pair_to_objs[pair] = []
      all_sub_rel_pair_to_objs[pair].append(link[2])
      
      if fold == 'train':
        if pair not in train_sub_rel_pair_to_objs:
          train_sub_rel_pair_to_objs[pair] = []
        train_sub_rel_pair_to_objs[pair].append(link[2])

  training_pairs = np.array(list(train_sub_rel_pair_to_objs.keys()))

  optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
  dataset_size = training_pairs.shape[0] # pylint: disable=E1136  # pylint/issues/3139
  for epoch in range(config.n_epochs):
    history = []
    np.random.shuffle(training_pairs)
    for batch_start in range(0, dataset_size, config.batch_size):
      batch = training_pairs[batch_start:batch_start + config.batch_size]
      real_batch_size = batch.shape[0]

      labels = np.zeros([real_batch_size, config.n_entities])
      for i in range(real_batch_size):
        labels[i, train_sub_rel_pair_to_objs[tuple(batch[i])]] = 1.0

      with tf.GradientTape() as tape:
          logits = model(batch, training=True)
          labels = tf.convert_to_tensor(labels, dtype=tf.float32)
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
    print(evaluation.evaluate(model, config, all_sub_rel_pair_to_objs, datasets['valid']))

def run_exp(config, datasets):
  models = [tucker.TuckerModel(config) for i in range(config.n_models)]
  for model in models:
    train_tucker(model, datasets, config)