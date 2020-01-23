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
    
    #link_ids.append((sub_idx, rel_idx, obj_idx))
    link_ids.append((obj_idx, rel_idx, sub_idx))
    
  return np.array(link_ids)

def get_dataset(data_dir):

  folds = ['train', 'valid', 'test', 'induced_all']
  
  entity2idx = {}
  rel2idx = {}
  dataset = {}

  for fold in folds:
    file_adrs = data_root+'/'+dataset_name+'/'+fold+'.txt'
    dataset[fold] = process_kb(open(file_adrs).readlines(), entity2idx, rel2idx)
