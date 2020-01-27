import numpy as np

def evaluate(model, config, all_sub_rel_pair_to_objs, dataset):
  ranks = []
  dataset_size = dataset.shape[0]
  
  for batch_start in range(0, dataset_size, config.batch_size):
    batch = dataset[batch_start:batch_start + config.batch_size]
    real_batch_size = batch.shape[0]
    
    labels = np.ones([real_batch_size, config.n_entities])
    for i in range(real_batch_size):
      pair = (batch[i,0], batch[i,1])
      labels[i, all_sub_rel_pair_to_objs[pair]] = 0.0
    
    logits = model(batch[:,:2], training=False).numpy()
      
    predicted_logit = logits[np.arange(real_batch_size), batch[:,2]]
    predicted_logit = np.expand_dims(predicted_logit, 1)
    compares = (logits>= predicted_logit)
    ranks.append(np.sum(compares*labels, 1) + 1) 
      
  ranks = np.concatenate(ranks)
  
  return {
      'R@1': np.mean(ranks==1),
      'R@3': np.mean(ranks<=3),
      'R@10': np.mean(ranks<=10),
      'MRR': np.mean(1.0/ranks)
  }