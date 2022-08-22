import torch
import warnings

class DefaultConfig(object):
    model = 'ResNet50_Siamese' # model name
    
    train_data_root1 = "the path of anchor samples of training set" # 'the path of anchor samples of training set'
    train_data_root2 = "the path of positive samples of training set" # 'the path of positive samples of training set'
    train_data_root3 = "the path of negative samples of training set" # 'the path of negative samples of training set'
    
    val_data_root1 = "the path of anchor samples of val set" # 'the path of anchor samples of val set'
    val_data_root2 = "the path of positive samples of val set" # 'the path of positive samples of val set' 
    val_data_root3 = "the path of negative samples val set" # 'the path of negative samples val set'
     
    test_data_root1 = "the path of anchor samples of testing set" # 'the path of anchor samples of testing set' 
    test_data_root2 = "the path of positive samples of testing set" # 'the path of positive samples of testing set' 
    test_data_root3 = "the path of negative samples testing set" # 'the path of negative samples testing set'
     
    load_train_model_path = "the path of training model" # 'the path of training model'
    load_test_model_path = "the path of testing model" # 'the path of testing model'
    
    
    train_batch_size = 64
    test_batch_size = 32
    scheduler = None
    num_workers = 8
    result_file = 'file to save the output of the model'  # 'file to save the output of the model'
    max_epoch = 20
    lr = 0.0001
    
    use_gpu = True
    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        opt.device = torch.device('cuda:0') if opt.use_gpu else torch.device('cpu')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
