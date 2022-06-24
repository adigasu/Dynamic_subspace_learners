from __future__ import print_function
from __future__ import division

import collections
import os
import matplotlib
import numpy as np
import logging
import torch
import time
import json
import random
import shelve
from tqdm import tqdm
import lib
from lib.clustering import make_clustered_dataloaders
import warnings

warnings.simplefilter("ignore", category=PendingDeprecationWarning)
os.putenv("OMP_NUM_THREADS", "8")


def load_config(config_name):
    with open(config_name, 'r') as f:
        config = json.load(f)

    # config = json.load(open(config_name))
    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                if type(config[k]) is str:
                    # if python types, then evaluate str expressions
                    if config[k][:5] in ['range', 'float']:
                        config[k] = eval(config[k])
            else:
                eval_json(config[k])

    eval_json(config)
    return config


def json_dumps(**kwargs):
    # __repr__ may contain `\n`, json replaces it by `\\n` + indent
    return json.dumps(**kwargs).replace('\\n', '\n    ')


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)


def evaluate(model, dataloaders, logging, backend='faiss', config=None):
    score = lib.utils.evaluate(
        model,
        dataloaders['eval'],
        use_penultimate=False,
        backend=backend
    )
    return score


def train_batch(model, criterion, opt, config, batch, dset, epoch):
    X = batch[0].cuda(non_blocking=True)  # images
    T = batch[1].cuda(non_blocking=True)  # class labels
    I = batch[2]  # image ids

    opt.zero_grad()
    M = model(X)

    if epoch >= config['finetune_epoch']:
        pass
    else:
        M = M.split(config['sz_embedding'] // config['nb_clusters'], dim=1)
        M = M[dset.id]

    M = torch.nn.functional.normalize(M, p=2, dim=1)
    loss = criterion[dset.id](M, T)
    loss.backward()
    opt.step()
    return loss.item()


def get_criterion(config):
    name = 'margin'
    ds_name = config['dataset_selected']
    nb_classes = len(
        config['dataset'][ds_name]['classes']['train']
    )
    logging.debug('Create margin loss. #classes={}'.format(nb_classes))
    criterion = [
        lib.loss.MarginLoss(
            nb_classes,
        ).cuda() for i in range(config['nb_clusters'])
    ]
    return criterion


def get_optimizer(config, model, criterion):
    opt = torch.optim.Adam([
        {
            'params': model.parameters_dict['backbone'],
            **config['opt']['backbone']
        },
        {
            'params': model.parameters_dict['embedding'],
            **config['opt']['embedding']
        }
    ])

    return opt


def start(config):
    import warnings

    metrics = {}

    # reserve GPU memory for faiss if faiss-gpu used
    faiss_reserver = lib.faissext.MemoryReserver()

    # model load
    load_epoch = '_' + str(config['load_epoch'])
    load_suff = load_epoch + '.pt'
    print("Load path: %s" %os.path.join(config['log']['path'], config['log']['name'] + load_suff))
    model_path= os.path.join(config['log']['path'], config['log']['name'] + load_suff)

    if not os.path.exists(model_path):
        warnings.warn('model_path file doesnot exists: {}'.format(_fpath))

    # warn if log file exists already and append underscore
    _fpath = os.path.join(config['log']['path'], config['log']['name'])
    if os.path.exists(_fpath):
        warnings.warn('Log file exists already: {}'.format(_fpath))
        print('Appending underscore to log file and database')
        config['log']['name'] += '_test_'

    # initialize logger
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG if config['verbose'] else logging.INFO,
        handlers=[
            logging.FileHandler(
                "{0}/{1}.log".format(
                    config['log']['path'],
                    config['log']['name']
                )
            ),
            logging.StreamHandler()
        ]
    )

    # print summary of config
    logging.info(
        json_dumps(obj=config, indent=4, cls=JSONEncoder, sort_keys=True)
    )

    torch.cuda.set_device(config['cuda_device'])

    if not os.path.isdir(config['log']['path']):
        os.mkdir(config['log']['path'])

    # set random seed for all gpus
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    faiss_reserver.lock(config['backend'])

    model = lib.model.make(config).cuda()

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # create eval dataloaders; init used for creating clustered DLs
    dataloaders = {}
    dataloaders['eval'] = lib.data.loader.make(config, model, 'eval')

    criterion = get_criterion(config)
    opt = get_optimizer(config, model, criterion)

    faiss_reserver.release()
    logging.info("Evaluating model...")
    metrics[-1] = {
        'score': evaluate(model, dataloaders, logging,
                          backend=config['backend'],
                          config=config)}
