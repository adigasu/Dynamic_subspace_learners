from __future__ import print_function
from __future__ import division

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from . import evaluation
from . import similarity
import numpy as np
import torch
import logging
from tqdm import tqdm


def predict_batchwise(model, dataloader, use_penultimate, is_dry_run=False):
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    # model.train()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():

        # use tqdm when the dataset is large (SOProducts)
        is_verbose = len(dataloader.dataset) > 0

        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc='predict', disable=not is_verbose):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    if not is_dry_run:
                        # move images to device of model (approximate device)
                        J = J.to(list(model.parameters())[0].device)
                        # predict model output for image
                        J = model(J, use_penultimate).data.cpu().numpy()
                        # grad_val = grad_val.data.cpu().numpy()
                        # take only subset of resulting embedding w.r.t dataset
                    else:
                        # just a placeholder not to break existing code
                        J = np.array([-1])
                for j in J:
                    A[i].append(np.asarray(j))
        result = [np.stack(A[i]) for i in range(len(A))]
    model.train()
    model.train(model_is_training)  # revert to previous training state
    if is_dry_run:
        # do not return features if is_dry_run
        return [None, *result[1:]]
    else:
        return result


def evaluate(model, dataloader, use_penultimate, backend,
             K=[1, 2, 4, 8], with_nmi=True):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader, use_penultimate)

    scores = {}

    # calculate NMI with kmeans clustering
    if with_nmi:
        nmi = evaluation.calc_normalized_mutual_information(
            T,
            similarity.cluster_by_kmeans(
                X, nb_classes, backend=backend
            )
        )
        logging.info("NMI: {:.3f}".format(nmi * 100))
        scores['nmi'] = nmi

    # get predictions by assigning nearest 8 neighbors with euclidian
    #assert np.max(K) <= 8, ("Sorry, this is hardcoded here."
    #                        " You would need to retrieve > 8 nearest neighbors"
    #                        " to calculate R@k with k > 8")
    Y = similarity.assign_by_euclidian_at_k(X, T, 8, backend=backend)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in K:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    scores['recall'] = recall

    return scores
