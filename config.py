def BIN_config_DBPE():
    config = {}
    config['batch_size'] = 16
    config['input_dim_drug'] = 23532
    config['input_dim_target'] = 16693
    config['train_epoch'] = 13
    config['max_drug_seq'] = 50
    config['max_protein_seq'] = 545
    config['emb_size'] = 384
    config['dropout_rate'] = 0.1
    
    #DenseNet
    config['scale_down_ratio'] = 0.25
    config['growth_rate'] = 20
    config['transition_rate'] = 0.5
    config['num_dense_blocks'] = 4
    config['kernal_dense_size'] = 3
    
    # Encoder
    config['intermediate_size'] = 1536
    config['num_attention_heads'] = 12
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    config['flat_dim'] = 78192
    return config