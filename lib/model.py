from math import ceil
import numpy as np
import os
import torchvision
import torch
from torch.nn import Linear, Dropout, AvgPool2d, MaxPool2d
from torch.nn import Conv2d, ReLU, Sigmoid, Softmax, BatchNorm2d
from torch.nn.modules.module import Module
from torch.nn.init import xavier_normal_
import logging


def resnet50(pretrained = True):
    model = torchvision.models.resnet50(pretrained = pretrained)

    model.features = torch.nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3 #, model.layer4
    )

    model.sz_features_output = 1024 #2048

    for module in filter(
        lambda m: type(m) == torch.nn.BatchNorm2d, model.modules()
    ):
        module.eval()
        module.train = lambda _: None

    return model


def make_parameters_dict(model, filter_module_names):
    """
    Separates model parameters into 'backbone' and other modules whose names
    are given in as list in `filter_module_names`, e.g. ['embedding_layer'].
    """

    # init parameters dict
    D = {k: [] for k in ['backbone', *filter_module_names]}
    for name, param in model.named_parameters():
        name = name.split('.')[0]
        if name not in filter_module_names:
            D['backbone'] += [param]
        else:
            D[name] += [param]

    # verify that D contains same number of parameters as in model
    nb_total = len(list(model.parameters()))
    nb_dict_params = sum([len(D[d]) for d in D])
    assert nb_total == nb_dict_params
    return D


def init_splitted(layer, nb_clusters, sz_embedding):
    # initialize splitted embedding parts separately
    from math import ceil
    for c in range(nb_clusters):
        i = torch.arange(
            c * ceil(sz_embedding / nb_clusters),
            # cut off remaining indices, e.g. if > embedding size
            min(
                (c + 1) * ceil(
                    sz_embedding / nb_clusters
                ),
                sz_embedding
            )
        ).long()
        _layer = torch.nn.Linear(layer.weight.shape[1], len(i))
        layer.weight.data[i] = xavier_normal_(_layer.weight.data, gain = 1)
        layer.bias.data[i] = _layer.bias.data


def split_neurons(layer, new_learner_neurons, r_learner_neurons, nb_learners, sz_embedding):
    # split the neuron and assign a new learner
    layer = layer.cpu()
    indices = torch.tensor(r_learner_neurons[:len(new_learner_neurons)]).long()
    indices_new_learner = torch.tensor(new_learner_neurons).long()
    for i in range(len(indices_new_learner)):
        layer.weight.data[indices[i]] = layer.weight.data[indices_new_learner[i]]
        layer.bias.data[indices[i]] = layer.bias.data[indices_new_learner[i]]

    reset_indices = torch.tensor(r_learner_neurons[len(indices_new_learner):]).long()

    # Reset remaining neurons
    _layer = torch.nn.Linear(layer.weight.shape[1], len(reset_indices))
    layer.weight.data[reset_indices] = xavier_normal_(_layer.weight.data, gain = 1)
    layer.bias.data[reset_indices] = _layer.bias.data
    nb_learners += 1
    layer = layer.cuda()

    # update sIndx, rIndx, nb_learners
    return indices.numpy(), reset_indices.numpy(), nb_learners


class FilterLearner:
    def __init__(self, model):
        self.model = model
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.filter_ranks = []
        self.filter_scores = []

    def reset(self, sz_embedding):
        self.filter_ranks = np.zeros(sz_embedding)
        self.filter_scores = np.zeros(sz_embedding)
        self.activations = []
        self.gradients = []
        self.grad_index = 0

    def save_gradients(self, grad):
        if torch.isnan(grad).any():
            logging.info("save_gradients: nan!!!")

        self.gradients.append(grad)
        self.grad_index +=1

    def normalize_ranks(self, score):
        # normalize v as in pruning
        v = torch.abs(score)
        v = v / torch.sqrt(torch.sum(v * v) + 1e-8)

        if torch.sum(torch.abs(v)) == 0:
            logging.info("normalize_ranks: zeros!!!")
        if torch.isnan(v).any():
            logging.info("normalize_ranks: nan!!!")

        # Normalize v in [0,1] or use softmax
        v = (v - v.min())/(v.max() - v.min())

        return v.cpu().numpy()

    def compute_score(self, r_learner_neurons):
        grads = []
        actns = []
        for i in range(self.grad_index):
            actns.extend(self.activations[i].data)
            grads.extend(self.gradients[i].data)

        self.activations = torch.stack(actns)
        self.gradients = torch.stack(grads)
        taylor_score = self.activations[:, r_learner_neurons] * self.gradients[:, r_learner_neurons]
        score = taylor_score.mean(dim = 0)
        self.filter_scores[r_learner_neurons] = self.normalize_ranks(score)
        self.filter_ranks = self.filter_scores.argsort()[::-1]

    def split_learner(self, r_learner_neurons):
        self.compute_score(r_learner_neurons)
        val = []
        indx = []
        for i in self.filter_ranks:
            print("self.filter_scores[i]: ", self.filter_scores[i], i)
            if self.filter_scores[i] > 0.5:
                val.append(self.filter_scores[i])
                indx.append(i)
            else:
                break

        if np.isnan(val).any():
            logging.info("split_learner: nan!!!")
            return []

        return indx.copy()


class AttentionLayer(Module):

    def __init__(self, input_dim, kernel_sz=3):
        super(AttentionLayer, self).__init__()

        self.layer1 = Conv2d(input_dim, input_dim//8, kernel_size=kernel_sz, stride=1, padding=1)
        self.activation1 = ReLU(inplace=True)

        self.layer2 = Conv2d(input_dim//8, input_dim//32, kernel_size=kernel_sz, stride=1, padding=1)
        self.activation2 = ReLU(inplace=True)

        self.layer3 = Conv2d(input_dim//32, 1, kernel_size=kernel_sz, stride=1, padding=1)
        self.Sigmoid = Sigmoid()

    def forward(self, x):
        B, N, W, H = x.size()

        l1 = self.layer1(x)
        a1 = self.activation1(l1)

        l2 = self.layer2(a1)
        a2 = self.activation2(l2)

        l3 = self.layer3(a2)
        out = self.Sigmoid(l3)

        return out


def embed_model(model, config, sz_embedding, normalize_output=True):

    # Attention model
    model.attention_map = AttentionLayer(model.sz_features_output)

    model.features_pooling = AvgPool2d(14,
        stride=1, padding=0, ceil_mode=True, count_include_pad=True
    )
    model.features_dropout = Dropout(0.01)

    # choose arbitrary parameter for selecting GPU/CPU
    dev = list(model.parameters())[0].device
    if type(model) != torchvision.models.ResNet:
        model.sz_features_output = _sz_features[type(model)]
    torch.random.manual_seed(config['random_seed'] + 1)
    model.embedding = Linear(model.sz_features_output, sz_embedding).to(dev)

    if config['dyn_learner'] == True:
        model.filter_learner = FilterLearner(model)

    # for fair comparison between different cluster sizes
    torch.random.manual_seed(config['random_seed'] + 1)
    np.random.seed(config['random_seed'] + 1)

    init_splitted(
        model.embedding, config['nb_clusters'], config['sz_embedding']
    )

    # features_parameters = model.features.parameters()

    model.parameters_dict = make_parameters_dict(
        model = model,
        filter_module_names = ['embedding']
    )

    assert normalize_output

    nb_clusters = config['nb_clusters']

    learner_neurons = [None] * nb_clusters
    for c in range(nb_clusters):
        learner_neurons[c] = np.arange(
            c * ceil(sz_embedding / nb_clusters),
            # cut off remaining indices, e.g. if > embedding size
            min(
                (c + 1) * ceil(
                    sz_embedding / nb_clusters
                ),
                sz_embedding
            )
        )
    model.learner_neurons = learner_neurons

    def forward(x, use_penultimate=False):
        x = model.features(x)
        A = model.attention_map(x)
        x = x * A
        x = model.features_pooling(x)
        x = model.features_dropout(x)
        x = x.view(x.size(0), -1)
        # grad_emb = torch.tensor([[0]])
        if not use_penultimate:
            x = model.embedding(x)
            for idxs in model.learner_neurons:
                x[:, idxs] = torch.nn.functional.normalize(
                    x[:, idxs], p=2, dim=1
                )
            if (model.training == True) and (config['dyn_learner'] == True):
                # for j in x.data.cpu().numpy():
                #     model.filter_learner.activations.append(j)
                if torch.sum(torch.abs(x)) == 0:
                    logging.info("activation: zeros!!!")
                if torch.isnan(x).any():
                    logging.info("activation: nan!!!")
                model.filter_learner.activations.append(x)
                x.register_hook(model.filter_learner.save_gradients)
            # get_gradients(x)
        else:
            # normalize the entire penultimate layer
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
    model.forward = forward


def make(config):
    model = resnet50(pretrained = True)
    embed_model(
        model = model,
        config = config,
        sz_embedding = config['sz_embedding'],
        normalize_output = True
    )
    return model
