import numpy as np
import tensorflow as tf
import os


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
            self.link_matrices[fold] = self._raw_links_to_matrix(self.raw_links[fold])

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
            omim2hpo_dict[omim_id] = np.array(omim2hpo_dict[omim_id])
        return omim2hpo_dict

    def _raw_links_to_matrix(self, raw_links):
        mat = np.zeros([len(self.omim2idx), len(self.hpo2idx)], dtype=np.float32)
        mat[raw_links[:, 0], raw_links[:, 1]] = 1.0
        return mat
