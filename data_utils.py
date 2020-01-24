import numpy as np

def process_kb(links, entity2idx, rel2idx):
  link_ids = []
  for link in links:
    sub, rel, obj = link.split()[:3]
    
    if sub not in entity2idx:
      entity2idx[sub] = len(entity2idx)
    if obj not in entity2idx:
      entity2idx[obj] = len(entity2idx)
    if rel not in rel2idx:
      rel2idx[rel] = len(rel2idx)
      
    sub_idx = entity2idx[sub]
    rel_idx = rel2idx[rel]
    obj_idx = entity2idx[obj]
    
    link_ids.append((obj_idx, rel_idx, sub_idx))
    
  return np.array(link_ids)

def create_datasets(data_dir):

  folds = ['train', 'valid', 'test']
  
  entity2idx = {}
  rel2idx = {}
  datasets = {}

  for fold in folds:
    file_adrs = data_dir + '/' + fold + '.txt'
    datasets[fold] = process_kb(open(file_adrs).readlines(), entity2idx, rel2idx)
  return datasets, entity2idx, rel2idx
