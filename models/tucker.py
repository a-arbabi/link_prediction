import tensorflow as tf

class TuckerModel(tf.keras.Model):
  def __init__(self, config):
    super(TuckerModel, self).__init__()

    self.config = config
    
    self.entity_embeddings = tf.Variable(
        tf.random.normal(
            [config.n_entities, config.d_entity],
            0.0, 0.1))  
    
    self.bias = tf.Variable(tf.zeros([config.n_entities,]))  
    
    ''' ##
    self.relation_embeddings = tf.Variable(
        tf.random.normal(
            [config.n_relations, config.d_relation],
            0.0, 0.1))  
    
    ''' ##
    self.core_tensor = tf.Variable(
        tf.random.normal(
            [config.d_relation, config.d_entity * config.d_entity],
            0.0, 0.1)) 
    
    self.input_dropout = tf.keras.layers.Dropout(config.input_d)
    self.hidden_dropout1 = tf.keras.layers.Dropout(config.hidden_d1)
    self.hidden_dropout2 = tf.keras.layers.Dropout(config.hidden_d2)

  def call(self, input, training=False):
    batch_sub_embedding = tf.nn.embedding_lookup(
        self.entity_embeddings, input[:,0])
    #batch_sub_embedding = tf.expand_dims(batch_sub_embedding, 1)
    batch_sub_embedding = self.input_dropout(
      batch_sub_embedding,
      training=training)
    
    '''
    batch_rel_embedding = tf.nn.embedding_lookup(
        self.relation_embeddings, input[:,1])
    w_mat = tf.matmul(batch_rel_embedding, self.core_tensor)
    w_mat = tf.reshape(w_mat, [-1, config.d_entity, config.d_entity])
    w_mat = self.hidden_dropout1(w_mat, training=training)
    '''
    ####
    w_mat = tf.reshape(self.core_tensor, [self.config.d_entity, self.config.d_entity])
    ####
    
    sub_rel_representation = tf.matmul(batch_sub_embedding, w_mat)
    sub_rel_representation = tf.squeeze(sub_rel_representation)
    sub_rel_representation = self.hidden_dropout2(
      sub_rel_representation, training=training)
    
    scores = tf.matmul(
      sub_rel_representation,
      self.entity_embeddings,
      transpose_b = True) + self.bias
    return scores