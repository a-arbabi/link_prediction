import numpy as np
import tensorflow as tf
import os


def process_kb(links, entity2idx, rel2idx, reverse=False):
  link_ids = []
  for link in links:
    sub, rel, obj = link.split()[:3]
    
    if reverse:
      rel = rel+'_reversed'
    if sub not in entity2idx:
      entity2idx[sub] = len(entity2idx)
    if obj not in entity2idx:
      entity2idx[obj] = len(entity2idx)
    if rel not in rel2idx:
      rel2idx[rel] = len(rel2idx)
    
    sub_idx = entity2idx[sub]
    rel_idx = rel2idx[rel]
    obj_idx = entity2idx[obj]
    
    if reverse:
      link_ids.append((obj_idx, rel_idx, sub_idx))
    else:
      link_ids.append((sub_idx, rel_idx, obj_idx))
    
  return np.array(link_ids)

def triplet_list_to_pair_dict(triplet_list, n_entities):
  sub_rel_pairs = {}
  for triplet in triplet_list:
    pair = (triplet[0], triplet[1])
    if pair not in sub_rel_pairs:
      sub_rel_pairs[pair] = set()
    sub_rel_pairs[pair].add(triplet[2])
  return sub_rel_pairs

def create_data_pair_list(data_path, omim2idx, hp2idx):
  link_ids = []
  for link in open(data_path).readlines():
    hp_term, rel, omim_term = link.split()[:3]
    del(rel)
    
    if hp_term not in hp2idx:
      hp2idx[hp_term] = len(hp2idx)
    if omim_term not in omim2idx:
      omim2idx[omim_term] = len(omim2idx)
    
    hp_idx = hp2idx[hp_term]
    omim_idx = omim2idx[omim_term]
    
    link_ids.append((omim_idx, hp_idx))

  return link_ids

def create_data_matrix(data_path, omim2idx, hp2idx):
  link_ids = create_data_pair_list(data_path, omim2idx, hp2idx)

  interaction_matrix = np.zeros((len(omim2idx), len(hp2idx)), dtype=np.float32)
  for link in link_ids:
    interaction_matrix[link[0], link[1]] = 1.0
    
  return interaction_matrix

def create_children_dict(data_path, term2idx):
  children_dict = {}
  for link in open(data_path).readlines():
    child_term, rel, parent_term = link.split()[:3]
    del(rel)
    
    if child_term not in term2idx:
      term2idx[child_term] = len(term2idx)
    if parent_term not in term2idx:
      term2idx[parent_term] = len(term2idx)
    
    child_idx = term2idx[child_term]
    parent_idx = term2idx[parent_term]

    if parent_idx not in children_dict: 
      children_dict[parent_idx] = set()
    children_dict[parent_idx].add(child_idx)

  return children_dict

def _back_dfs(term, parents_dict, mark , ancestry_matrix):
  if term in mark:
    return
  mark.add(term)
  ancestry_matrix[term, term] = 1.0
  if term not in parents_dict:
    return
  num_parents = len(parents_dict[term])
  for parent in parents_dict[term]:
    _back_dfs(parent, parents_dict, mark, ancestry_matrix)
    ancestry_matrix[term] += ancestry_matrix[parent]/num_parents




def create_weighted_ancestry_matrix(children_dict, term2idx):
  parents_dict = {}
  for parent in children_dict:
    for child in children_dict[parent]:
      if child not in parents_dict:
        parents_dict[child] = set()
      parents_dict[child].add(parent)
  mark = set()
  ancestry_matrix = np.zeros((len(term2idx), len(term2idx)), dtype=np.float32)
  for term in term2idx:
    _back_dfs(term2idx[term], parents_dict, mark, ancestry_matrix)
  return ancestry_matrix




def create_aggregated_dataset(
  triplet_list, n_entities, ground_truth_triplets=None):
  sub_rel_pairs = triplet_list_to_pair_dict(triplet_list, n_entities)

  if ground_truth_triplets is not None:
    gt_sub_rel_pairs = triplet_list_to_pair_dict(
      ground_truth_triplets, n_entities)

  def gen():
    for pair in sub_rel_pairs:
      data_dict = {
        'sub':pair[0],
        'rel':pair[1],
        'obj_list':list(sub_rel_pairs[pair])
        }
      if ground_truth_triplets is not None:
        data_dict['gt_obj_list'] = list(gt_sub_rel_pairs[pair])
      yield data_dict

  tensor_types = {
    'sub': tf.int64,
    'rel': tf.int64,
    'obj_list': tf.int64,
  }
  tensor_shapes = {
    'sub': tf.TensorShape([]),
    'rel': tf.TensorShape([]),
    'obj_list': tf.TensorShape([None]),
  }
  if ground_truth_triplets is not None:
    tensor_types['gt_obj_list'] = tf.int64
    tensor_shapes['gt_obj_list'] = tf.TensorShape([None])

  dataset = tf.data.Dataset.from_generator(
    gen,
    tensor_types,
    tensor_shapes
  )
  def _make_dense(x):
    output = {
      'sub': x['sub'],
      'rel': x['rel'],
      'obj_list': tf.scatter_nd(
        tf.expand_dims(x['obj_list'], 1),
        tf.ones_like(x['obj_list'], dtype=tf.float32),
        [n_entities]),
    }
    if ground_truth_triplets is not None:
      output['gt_obj_list'] = tf.scatter_nd(
        tf.expand_dims(x['gt_obj_list'], 1),
        tf.ones_like(x['gt_obj_list'], dtype=tf.float32),
        [n_entities])
    return output

  dataset = dataset.map(_make_dense)

  return dataset

def hpo_dfs(hp_idx, children, hp_descendants):
  if hp_idx in hp_descendants:
    return
  hp_descendants[hp_idx] = set()

  if hp_idx not in children:
    return
  for child_idx in children[hp_idx]:
    hpo_dfs(child_idx, children, hp_descendants)
    hp_descendants[hp_idx].add(child_idx)
    hp_descendants[hp_idx].add(child_idx)
    for hp_desc in hp_descendants[child_idx]:
      hp_descendants[hp_idx].add(hp_desc)
  
def create_datasets(data_dir):

  folds = ['train', 'valid', 'test']
  
  entity2idx = {}
  rel2idx = {}
  datasets = {}

  for fold in folds:
    file_adrs = data_dir + '/' + fold + '.txt'
    datasets[fold] = process_kb(
      open(file_adrs).readlines(), entity2idx, rel2idx, reverse=True)
  # dataset[fold].shape = [?, 3]
  # dataset[fold][:] = [omim, 0, hpo]
  
  hypernyms_path = os.path.join(data_dir, 'hpo_hypernyms.txt')
  hypernyms = process_kb(
    open(hypernyms_path).readlines(), entity2idx, rel2idx)
  # hypernyms.shape = [?, 3]
  # dataset[:] = [child, 1, parent]

  children = {} 
  parents = {}
  for link in hypernyms:
    if link[2] not in children:
      children[link[2]] = set()
    children[link[2]].add(link[0])
    if link[0] not in parents:
      parents[link[0]] = set()
    parents[link[0]].add(link[2])

  hp_descendants = {}

  hpo_root = entity2idx['HP:0000118']
  hpo_dfs(hpo_root, children, hp_descendants)

  hp_ancestors = {}
  for hp_idx in hp_descendants:
    for desc in hp_descendants[hp_idx]:
      if desc not in hp_ancestors:
        hp_ancestors[desc] = set()
      hp_ancestors[desc].add(hp_idx)

  omim2hpo_direct = {}
  for fold in folds:
    omim2hpo_direct[fold] = {}
    for link in datasets[fold]:
      if link[0] not in omim2hpo_direct[fold]:
        omim2hpo_direct[fold][link[0]] = set()
      omim2hpo_direct[fold][link[0]].add(link[2])

  omim2hpo_induced = {}
  omim2hpo_induced_combined = {}
  for fold in folds:
    omim2hpo_induced[fold] = {}
    for omim in omim2hpo_direct[fold]:
      if omim not in omim2hpo_induced_combined:
        omim2hpo_induced_combined[omim] = set()
      if omim not in omim2hpo_induced[fold]:
        omim2hpo_induced[fold][omim] = set()
      for hp_idx in omim2hpo_direct[fold][omim]:
        for hp_anc in hp_ancestors[hp_idx]:
          if hp_anc not in omim2hpo_induced_combined[omim]:
            omim2hpo_induced_combined[omim].add(hp_anc)
            omim2hpo_induced[fold][omim].add(hp_anc)

  rel2idx['is_manifested_in_reversed_induced'] = len(rel2idx)
  rel2idx['is_a_induced'] = len(rel2idx)

  omim_induced = {}
  for fold in folds:
    omim_induced[fold] = []
    for omim in omim2hpo_induced[fold]:
      for hp_idx in omim2hpo_induced[fold][omim]:
        omim_induced[fold].append(
          [omim, rel2idx['is_manifested_in_reversed_induced'], hp_idx])
    omim_induced[fold] = np.array(omim_induced[fold])


  is_a_induced = []
  for hp_idx in hp_ancestors:
    for hp_anc in hp_ancestors[hp_idx]:
      is_a_induced.append([hp_idx, rel2idx['is_a_induced'], hp_anc])
  is_a_induced = np.array(is_a_induced)
  sparse_anc_matrix = tf.SparseTensor(
    is_a_induced[:,[0,2]], np.ones(shape=[is_a_induced.shape[0]]), # pylint: disable=E1136  # pylint/issues/3139 
    [len(entity2idx), len(entity2idx)])

  for fold in folds:
    datasets[fold] = np.concatenate([datasets[fold], omim_induced[fold]])
  datasets['train'] = np.concatenate([datasets['train'], hypernyms])
  datasets['train'] = np.concatenate([datasets['train'], is_a_induced])

  return datasets, entity2idx, rel2idx, sparse_anc_matrix

def prepare_matrix_datasets(data_dir, omim2idx, hp2idx):
  hypernyms_path = os.path.join(data_dir, 'hpo_hypernyms.txt')
  children_dict = create_children_dict(hypernyms_path, hp2idx)
  ancestry_matrix = create_weighted_ancestry_matrix(children_dict, hp2idx)

  interaction_matrix = {}
  for fold in ['train', 'valid', 'test']:
    data_path = os.path.join(data_dir, fold + '.txt')
    interaction_matrix[fold] = create_data_matrix(data_path, omim2idx, hp2idx)

  for fold in ['train', 'valid', 'test']:
    mat_copy = interaction_matrix[fold]
    interaction_matrix[fold] = np.zeros(
      [len(omim2idx), len(hp2idx)], dtype=interaction_matrix[fold].dtype)
    interaction_matrix[fold][:mat_copy.shape[0], :mat_copy.shape[1]] = mat_copy

  gt_matrix = np.sum(list(interaction_matrix.values()), axis=0)
  gt_matrix = np.matmul(gt_matrix, ancestry_matrix)
  binary_gt_matrix = (gt_matrix>0) * 1.0

  return interaction_matrix, binary_gt_matrix

