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
        # Dynamic learner or baseline method
        if config['dyn_learner'] == True:
            split_array = [len(i) for i in model.learner_neurons]
            M = M.split(split_array, dim=1)
        else:
            # Baseline (DCML) method
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
    metrics = {}

    # reserve GPU memory for faiss if faiss-gpu used
    faiss_reserver = lib.faissext.MemoryReserver()

    # create logging directory
    os.makedirs(config['log']['path'], exist_ok=True)

    # warn if log file exists already and append underscore
    import warnings
    _fpath = os.path.join(config['log']['path'], config['log']['name'])
    if os.path.exists(_fpath):
        warnings.warn('Log file exists already: {}'.format(_fpath))
        print('Appending underscore to log file and database')
        config['log']['name'] += '_'

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

    start_epoch = 0
    best_epoch = -1
    best_recall = 0
    best_nmi = 0
    explore_dK = True

    # create init and eval dataloaders; init used for creating clustered DLs
    dataloaders = {}
    for dl_type in ['init', 'eval']:
        dataloaders[dl_type] = lib.data.loader.make(config, model, dl_type)

    criterion = get_criterion(config)
    opt = get_optimizer(config, model, criterion)

    faiss_reserver.release()
    logging.info("Evaluating initial model...")
    metrics[-1] = {
        'score': evaluate(model, dataloaders, logging,
                          backend=config['backend'],
                          config=config)}

    dataloaders['train'], C, T, I = make_clustered_dataloaders(model,
                                                               dataloaders['init'], config, reassign=False,
                                                               logging=logging)
    faiss_reserver.lock(config['backend'])

    metrics[-1].update({'C': C, 'T': T, 'I': I})

    logging.info("Training for {} epochs.".format(config['nb_epochs']))
    losses = []
    t1 = time.time()

    for e in range(start_epoch, config['nb_epochs']):
        is_best = False

        metrics[e] = {}
        time_per_epoch_1 = time.time()
        losses_per_epoch = []

        if e >= config['finetune_epoch']:
            if e == config['finetune_epoch'] or e == start_epoch:
                logging.info('Starting to finetune model...')
                config['nb_clusters'] = 1
                logging.debug(
                    "config['nb_clusters']: {})".format(config['nb_clusters']))
                faiss_reserver.release()
                dataloaders['train'], C, T, I = make_clustered_dataloaders(
                    model, dataloaders['init'], config, logging=logging)
                assert len(dataloaders['train']) == 1
        elif e > 0 and config['recluster']['enabled'] and \
                config['nb_clusters'] > 0:
            if e % config['recluster']['mod_epoch'] == 0:
                logging.info("Reclustering dataloaders...")
                faiss_reserver.release()
                dataloaders['train'], C, T, I = make_clustered_dataloaders(
                    model, dataloaders['init'], config, reassign=True,
                    C_prev=C, I_prev=I, logging=logging)
                faiss_reserver.lock(config['backend'])
                if config['verbose']:
                    for c in range(config['nb_clusters']):
                        print(
                            np.bincount(
                                np.array(
                                    dataloaders['train'][c].dataset.ys)
                            )[:200]
                        )

                metrics[e].update({'C': C, 'T': T, 'I': I})

        # merge dataloaders (created from clusters) into one dataloader
        mdl = lib.data.loader.merge(dataloaders['train'])

        # calculate number of batches for tqdm
        max_len_dataloaders = max([len(dl) for dl in dataloaders['train']])
        num_batches_approx = max_len_dataloaders * len(dataloaders['train'])

        if config['dyn_learner'] == True:
            model.filter_learner.reset(config['sz_embedding'])
            
        for batch, dset in tqdm(
                mdl,
                total=num_batches_approx,
                disable=num_batches_approx < 100,
                desc='Train epoch {}.'.format(e)
        ):
        # for batch, dset in tqdm(mdl, disable=True, desc='Train epoch {}.'.format(e)):
            loss = train_batch(model, criterion, opt, config, batch, dset, e)
            losses_per_epoch.append(loss)

        time_per_epoch_2 = time.time()
        losses.append(np.mean(losses_per_epoch[-20:]))
        logging.info(
            "Epoch: {}, loss: {}, time (seconds): {:.2f}.".format(
                e,
                losses[-1],
                time_per_epoch_2 - time_per_epoch_1
            )
        )

        faiss_reserver.release()
        tic = time.time()
        metrics[e].update({
            'score': evaluate(model, dataloaders, logging,
                              backend=config['backend'],
                              config=config),
            'loss': {
                'train': losses[-1]
            }
        })
        logging.debug(
            'Evaluation total elapsed time: {:.2f} s'.format(
                time.time() - tic
            )
        )
        faiss_reserver.lock(config['backend'])

        # recall_curr = metrics[e]['score']['recall'][0]  # take R@1
        nmi_curr = metrics[e]['score']['nmi']  # take R@1
        if nmi_curr > best_nmi:
            best_nmi = nmi_curr
            best_epoch = e
            is_best = True
            explore_dK = True
            logging.info('Best epoch!')

        model.current_epoch = e

        # save metrics etc. to shelve file
        with shelve.open(
                os.path.join(
                    config['log']['path'], config['log']['name']),
                writeback=True
        ) as _f:
            if 'config' not in _f:
                _f['config'] = config
            if 'metrics' not in _f:
                _f['metrics'] = {}
                # if initial model evaluated, append metrics
                if -1 in metrics:
                    _f['metrics'][-1] = metrics[-1]
            _f['metrics'][e] = metrics[e]

        if config['save_model'] and is_best:
            save_suff = '_' + str(e) + '.pt'
            torch.save(
                model.state_dict(),
                os.path.join(
                    config['log']['path'], config['log']['name'] + save_suff
                )
            )
            logging.info('Save the checkpoint!')


        # Dynamic learner (best epoch strategie)
        if config['dyn_learner'] == True and e > 0 and (e >= (best_epoch + 10)) and explore_dK and e < config['finetune_epoch']: 
            logging.info("Epoch: {}, nb_learners: {}. Exploring new learner........!!!".format(e, config['nb_clusters']))

            if model.filter_learner.grad_index == 0:
                logging.info("No grad_index to compute score!!!")

            elif len(model.learner_neurons[-1]) > 10:
                # model.filter_learner.compute_rank()
                new_learner_neurons = model.filter_learner.split_learner(model.learner_neurons[-1])

                if len(new_learner_neurons) >= 5:
                    # split the neurons
                    logging.info("Split the neurons to new learner of length {}".format(len(new_learner_neurons)))
                    new_learner_indices, reset_indices, nb_learners = lib.model.split_neurons(model.embedding, new_learner_neurons, model.learner_neurons[-1], config['nb_clusters'], config['sz_embedding'])

                    # update the model.learner_neurons
                    model.learner_neurons[-1] = new_learner_indices
                    model.learner_neurons.append(reset_indices)
                    logging.info("Epoch {}: {} learner and subspace index are {}".format(e, nb_learners, model.learner_neurons))

                    if nb_learners > 1:
                        config['recluster']['enabled'] = True
                    config['nb_clusters'] = nb_learners

                    # assign new criteria and optimizer
                    criterion = get_criterion(config)
                    explore_dK = False
                else:
                    logging.info("**** Not enough neurons for learner (new_learner_neurons {})".format(len(new_learner_neurons)))
            else:
                logging.info("!!!!! Not enough neurons to explore subspace")


    t2 = time.time()
    logging.info(
        "Total training time (minutes): {:.2f}.".format(
            (t2 - t1) / 60
        )
    )
    logging.info("Best nmi = {} at epoch {}.".format(best_nmi, best_epoch))
