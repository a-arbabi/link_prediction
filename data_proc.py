import numpy as np
import tensorflow as tf
import os
from collections import namedtuple


class OmimHpoData:
    def __init__(self, dataset_dir):
        self.omim2idx = {}
        self.hpo2idx = {}

        self.raw_links = {}
        self.omim2hpo_dicts = {}
        self.link_matrices = {}

        for fold in ["train", "valid", "test"]:
            file_name = fold + ".txt"
            file_path = os.path.join(dataset_dir, file_name)
            self.raw_links[fold] = self._process_links_file(file_path)

        self.raw_links["ground_truth"] = np.concatenate(
            list(self.raw_links.values()), axis=0
        )

        for fold in self.raw_links.keys():
            self.omim2hpo_dicts[fold] = self._raw_links_to_dict(self.raw_links[fold])

        hypernyms_path = os.path.join(dataset_dir, "hpo_hypernyms.txt")
        self.hpo2children, self.hpo2parents = self._process_hpo_hypernyms_file(
            hypernyms_path
        )
        self.weighted_ancestry_matrix = self._create_weighted_ancestry_matrix(
            self.hpo2parents
        )
        self.binary_ancestry_matrix = (self.weighted_ancestry_matrix > 0).astype(
            dtype=np.float32
        )

        for fold in self.raw_links.keys():
            self.link_matrices[fold] = self._raw_links_to_matrix(self.raw_links[fold])

        SpecialTokens = namedtuple("SpecialTokens", ["aggregate", "padding"])
        self.special_tokens = SpecialTokens(
            aggregate=len(self.hpo2idx), padding=len(self.hpo2idx)+1
        )

    def _process_hpo_hypernyms_file(self, file_path):
        children_dict = {}
        parents_dict = {}
        term2idx = self.hpo2idx

        for link in open(file_path).readlines():
            child_term, rel, parent_term = link.split()[:3]
            del rel

            if child_term not in term2idx:
                term2idx[child_term] = len(term2idx)
            if parent_term not in term2idx:
                term2idx[parent_term] = len(term2idx)

            child_idx = term2idx[child_term]
            parent_idx = term2idx[parent_term]

            if parent_idx not in children_dict:
                children_dict[parent_idx] = set()
            children_dict[parent_idx].add(child_idx)

            if child_idx not in parents_dict:
                parents_dict[child_idx] = set()
            parents_dict[child_idx].add(parent_idx)

        return children_dict, parents_dict

    def _create_weighted_ancestry_matrix(self, parents_dict):
        term2idx = self.hpo2idx
        ancestry_matrix = np.zeros((len(term2idx), len(term2idx)), dtype=np.float32)

        def _back_dfs(term):
            if ancestry_matrix[term, term] > 0:
                return
            ancestry_matrix[term, term] = 1.0
            if term not in parents_dict:
                return
            num_parents = len(parents_dict[term])
            for parent in parents_dict[term]:
                _back_dfs(parent)
                ancestry_matrix[term] += ancestry_matrix[parent] / num_parents

        for term in term2idx:
            _back_dfs(term2idx[term])
        return ancestry_matrix

    def _process_links_file(self, file_path):
        link_ids = []
        for link_str in open(file_path).readlines():
            hpo, rel, omim = link_str.split()[:3]

            if omim not in self.omim2idx:
                self.omim2idx[omim] = len(self.omim2idx)
            if hpo not in self.hpo2idx:
                self.hpo2idx[hpo] = len(self.hpo2idx)

            omim_idx = self.omim2idx[omim]
            hpo_idx = self.hpo2idx[hpo]

            link_ids.append((omim_idx, hpo_idx))
        return np.array(link_ids)

    def _raw_links_to_dict(self, raw_links):
        omim2hpo_dict = {}
        for link in raw_links:
            omim_id, hpo_id = link
            if omim_id not in omim2hpo_dict:
                omim2hpo_dict[omim_id] = set()
            omim2hpo_dict[omim_id].add(hpo_id)
        for omim_id in omim2hpo_dict:
            omim2hpo_dict[omim_id] = np.array(list(omim2hpo_dict[omim_id]))
        return omim2hpo_dict

    def _raw_links_to_matrix(self, raw_links):
        mat = np.zeros([len(self.omim2idx), len(self.hpo2idx)], dtype=np.float32)
        mat[raw_links[:, 0], raw_links[:, 1]] = 1.0
        return mat


def omim2hpo_dict_to_tf_dataset(
    omim2hpo_dict,
    special_tokens,
    drop_sizes=[0.0],
    n_repeats=1,
    batch_size=None,
    hpo_size=None,
):
    def gen():
        for omim in omim2hpo_dict:
            hpo_labels = list(omim2hpo_dict[omim])
            binary_labels = np.zeros([hpo_size], dtype=np.float32)
            binary_labels[hpo_labels] = 1.0
            for drop_size in drop_sizes:
                for rep in range(n_repeats):
                    hpo_inputs = np.random.permutation(hpo_labels)
                    hpo_inputs = hpo_inputs[int(len(hpo_inputs) * drop_size) :]
                    hpo_inputs = np.append([special_tokens.aggregate], hpo_inputs)
                    data_dict = {
                        "omim_idx": omim,
                        "hpo_labels": hpo_labels,
                        "hpo_inputs": hpo_inputs,
                        "binary_labels": binary_labels,
                    }
                    yield data_dict

    tensor_types = {
        "omim_idx": tf.int64,
        "hpo_labels": tf.int64,
        "hpo_inputs": tf.int64,
        "binary_labels": tf.float32,
    }
    tensor_shapes = {
        "omim_idx": tf.TensorShape([]),
        "hpo_labels": tf.TensorShape([None]),
        "hpo_inputs": tf.TensorShape([None]),
        "binary_labels": tf.TensorShape([hpo_size]),
    }
    dataset = tf.data.Dataset.from_generator(gen, tensor_types, tensor_shapes)
    dataset = dataset.shuffle(1000)
    if batch_size is not None:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes={
                "omim_idx": [],
                "hpo_labels": [None],
                "hpo_inputs": [None],
                "binary_labels": [hpo_size],
            },
            padding_values={
                "omim_idx": tf.cast(special_tokens.padding, dtype=tf.int64),
                "hpo_labels": tf.cast(special_tokens.padding, tf.int64),
                "hpo_inputs": tf.cast(special_tokens.padding, tf.int64),
                "binary_labels": tf.cast(special_tokens.padding, tf.float32),
            },
        )
    return dataset


def create_input_dataset_for_inference(omim_list, data_reader, batch_size):
    input_dict = {}
    for omim in omim_list:
        if omim in data_reader.omim2hpo_dicts["train"]:
            input_dict[omim] = data_reader.omim2hpo_dicts["train"][omim]
        else:
            input_dict[omim] = np.array([])

    input_dataset = omim2hpo_dict_to_tf_dataset(
        input_dict,
        special_tokens=data_reader.special_tokens,
        batch_size=batch_size,
        hpo_size=len(data_reader.hpo2idx),
        n_repeats=1,
        drop_sizes=[0.0],
    )
    return input_dataset

