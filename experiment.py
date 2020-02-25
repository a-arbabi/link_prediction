import models.tucker as tucker
from models import ease, trec
import numpy as np
import tensorflow as tf
import evaluation
import data_utils
import data_proc
import sys
import os


class exp_params:
    use_ncr = False
    model = "tucker"
    consider_hypernymy_as_relation = False
    sum_decendents = False
    loss_type = "cross entropy softmax"


def print_results(results, output_file=None):
    if output_file is None:
        output_file = sys.stdout
    for cat in results:
        print(cat, ":", results[cat].numpy(), end="\t", file=output_file)
    print("", file=output_file)


def train_tucker(model, train_dataset, valid_dataset, config):
    @tf.function
    def apply_batch(model, batch, config, optimizer):
        labels = batch["obj_list"]

        with tf.GradientTape() as tape:
            logits = model(batch, training=True)
            n_hits = tf.reduce_sum(labels, axis=1, keepdims=True)
            labels = labels / n_hits
            if config.use_softmax:
                loss = tf.reduce_mean(
                    n_hits
                    * tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits
                    )
                )
            else:
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels, logits=logits
                    )
                )

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
        if (epoch + 1) % 5 == 0:
            valid_results = evaluation.evaluate(model, config, valid_dataset)
            if len(best_results) == 0 or best_results["MRR"] < valid_results["MRR"]:
                best_results = valid_results
                best_weights = model.get_weights()
            print_results(valid_results)
    model.set_weights(best_weights)
    return best_results


def eval_trec_model(model, valid_input_dataset, data_reader):
    model_ans = np.zeros_like(data_reader.link_matrices["train"])
    for i, batch in enumerate(valid_input_dataset):
        x = batch["hpo_inputs"]
        mask = create_padding_mask(x, data_reader.special_tokens.padding)
        logits = model(x, False, mask=mask)
        model_ans[batch["omim_idx"]] = logits
    return evaluation.evaluate_by_matrix(
        model_ans,
        data_reader.link_matrices["ground_truth"],
        data_reader.raw_links["valid"],
    )


def create_padding_mask(seq, padding_token):
    seq = tf.cast(tf.math.equal(seq, padding_token), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def train_trec(model, data_reader, config):
    train_dataset = data_proc.omim2hpo_dict_to_tf_dataset(
        data_reader.omim2hpo_dicts["train"],
        special_tokens=data_reader.special_tokens,
        batch_size=config.batch_size,
        hpo_size=len(data_reader.hpo2idx),
        drop_sizes=[0.0, 0.25, 0.50, 0.75],
    )
    train_dataset = train_dataset.cache()
    valid_input_dataset = data_proc.create_input_dataset_for_inference(
        data_reader.omim2hpo_dicts["valid"].keys(), data_reader, config.batch_size
    )
    valid_input_dataset = valid_input_dataset.cache()

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.int64),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        ]
    )
    def apply_batch(x, labels):
        with tf.GradientTape() as tape:
            mask = create_padding_mask(x, data_reader.special_tokens.padding)
            logits = model(x, training=True, mask=mask)
            n_hits = tf.reduce_sum(labels, axis=1, keepdims=True)
            labels = labels / n_hits
            loss = tf.reduce_mean(
                n_hits
                * tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best_results = {}
    best_weights = None
    for epoch in range(config.n_epochs):
        history = []
        for i, batch in enumerate(train_dataset):
            loss = apply_batch(batch["hpo_inputs"], batch["binary_labels"])
            history.append(loss.numpy())

        print("Epoch::", epoch, ", Loss::", np.mean(history))
        if (epoch + 1) % 5 == 0:
            valid_results = eval_trec_model(model, valid_input_dataset, data_reader)
            if len(best_results) == 0 or best_results["MRR"] < valid_results["MRR"]:
                best_results = valid_results
                best_weights = model.get_weights()
            print_results(valid_results)
    # model.set_weights(best_weights)
    return best_results


def run_ease_exp(dataset_path, lambda_param):
    omim2idx = {}
    hp2idx = {}
    valid_path = os.path.join(dataset_path, "valid.txt")
    valid_pairs = data_utils.create_data_pair_list(valid_path, omim2idx, hp2idx)
    interaction_matrix, binary_gt_matrix = data_utils.prepare_matrix_datasets(
        dataset_path, omim2idx, hp2idx
    )

    train_matrix = interaction_matrix["train"]
    ans, model = ease.train_model(train_matrix, lambda_param)
    results = evaluation.evaluate_by_matrix(ans, binary_gt_matrix, valid_pairs)
    print_results(results)
    return model, results


def run_exp(exp_params, config, dataset_path):
    triplets, entity2idx, rel2idx, sp_mat = data_utils.create_datasets(dataset_path)
    del sp_mat
    gt_triplets = np.concatenate(
        [triplets[fold] for fold in ["train", "valid", "test"]]
    )
    config.n_entities = len(entity2idx)
    config.n_relations = len(rel2idx)

    def filter_fn(x):
        return tf.equal(x["rel"], 0)

    train_dataset = data_utils.create_aggregated_dataset(
        triplets["train"], config.n_entities
    )
    train_dataset = (
        train_dataset.filter(filter_fn).shuffle(500).batch(config.batch_size)
    ).cache()

    valid_dataset = data_utils.create_aggregated_dataset(
        triplets["valid"], config.n_entities, gt_triplets
    )
    valid_dataset = valid_dataset.filter(filter_fn).batch(config.batch_size)

    """
  models = [tucker.TuckerModel(config) for i in range(config.n_models)]
  for model in models:
    train_tucker(model, train_dataset, valid_dataset, config)
  """

    model = tucker.TuckerModel(config)
    best_result = train_tucker(model, train_dataset, valid_dataset, config)
    print_results(best_result)
    return model, best_result


#  return models[-1]
def trec_grid_search(input_path, output_path):
    class learning_config:
        lr = 0.0002
        n_epochs = 200
        batch_size = 1024

    class model_config:
        num_layers = 3
        d_model = 256
        num_heads = 4
        dff = 256

    data_reader = data_proc.OmimHpoData("../hpo_dataset")

    with open(output_path, "w") as output_file:
        for model_config.num_layers in [2, 4, 8]:
            for model_config.dff in [256, 512, 1024]:
                for model_config.num_heads in [4, 8]:
                    model = trec.TrecModel(model_config, len(data_reader.hpo2idx))
                    best_result = train_trec(model, data_reader, learning_config)
                    # model, best_result = run_exp(exp_params, config, dataset_path)
                    del model
                    output_file.write("\tnum_layers: " + str(model_config.num_layers))
                    output_file.write("\tdff: " + str(model_config.dff))
                    output_file.write("\tnum_heads: " + str(model_config.num_heads))
                    output_file.write("\t")
                    print_results(best_result, output_file)
                    output_file.flush()


def grid_search_exp(exp_params, dataset_path, output_path):
    config = tucker.config

    config.use_core_tensor = False
    with open(output_path, "w") as output_file:
        # for config.use_core_tensor in [True]:
        for config.lr in [0.001, 0.002]:
            for config.batch_size in [128, 256]:
                for config.d_entity in [100, 250, 500]:
                    model, best_result = run_exp(exp_params, config, dataset_path)
                    del model
                    #          output_file.write('use_core_tensor: ' + str(config.use_core_tensor))
                    output_file.write("\tlr: " + str(config.lr))
                    output_file.write("\tbatch_size: " + str(config.batch_size))
                    output_file.write("\td_entity: " + str(config.d_entity))
                    output_file.write("\t")
                    print_results(best_result, output_file)
                    output_file.flush()


def main():
    """
    grid_search_exp(
        exp_params, "../hpo_dataset", "../grid_search_result_no_core_tensor.txt"
    )
    """
    trec_grid_search("../hpo_dataset", "../trec_grid_search.txt")
    # run_exp(exp_params, tucker.config, '../hpo_dataset')


if __name__ == "__main__":
    main()

