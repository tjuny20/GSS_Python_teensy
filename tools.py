import numpy as np
import torch
import torch.nn as nn
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.decomposition import PCA
from torch.linalg import inv, eig, pinv


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


activation = SurrGradSpike.apply

class LIF_neuron(nn.Module):
    LIFstate = namedtuple('LIFstate', ['syn', 'mem', 'S'])

    def __init__(self, nb_inputs, nb_outputs, alpha, beta, is_recurrent=True, fwd_weight_scale=1.0, mean_fwd_weight=0.0,
                 rec_weight_scale=1.0, is_trained=True, layer=None, thr=1, vmin=-10):
        super(LIF_neuron, self).__init__()

        if layer == 0:
            diag_weights = torch.ones((nb_inputs,))     # Setting w. If is the input layer, w is just the identity
            self.weight = torch.nn.Parameter(torch.diag(diag_weights), requires_grad=is_trained)
        else:                     # Otherwise is initialized using nn.init.normal_: Values drawn from normal(mean, std)
            self.weight = torch.nn.Parameter(torch.empty((nb_inputs, nb_outputs)), requires_grad=is_trained)
            torch.nn.init.normal_(self.weight, mean=mean_fwd_weight, std=fwd_weight_scale / np.sqrt(nb_inputs)) # He initialization

        self.offset = torch.nn.Parameter(torch.empty((nb_outputs,)), requires_grad=is_trained) # Setting biases.
        torch.nn.init.normal_(self.offset, mean=0, std=1 / np.sqrt(nb_outputs))

        self.is_recurrent = is_recurrent
        if is_recurrent:
            # weight_rec = torch.empty((nb_outputs, nb_outputs))
            # torch.nn.init.normal_(weight_rec, mean=0.0, std=rec_weight_scale / np.sqrt(nb_outputs))
            self.weight_rec = torch.nn.Parameter(torch.empty((nb_outputs, nb_outputs)), requires_grad=True)
            # torch.nn.init.zeros_(self.weight_rec)
            torch.nn.init.normal_(self.weight_rec, mean=0.0, std=rec_weight_scale / np.sqrt(nb_inputs))

        self.alpha = alpha
        self.beta = beta
        self.state = None
        self.thr = thr
        self.vmin = vmin
        self.layer = layer

    def initialize(self, input):
        self.state = self.LIFstate(syn=torch.zeros_like(input, device=input.device),
                                   mem=torch.zeros_like(input, device=input.device),
                                   S=torch.zeros_like(input, device=input.device))

    def reset(self):
        self.state = None

    def forward(self, input):
        # h1 = torch.mm(input, self.weight)
        bs = input.shape[0]
        h1 = torch.mm(input, self.weight) + self.offset.repeat(bs, 1) * ADD_OFFSET_LIF  # WHY use repeat?

        if self.state is None:
            self.initialize(h1) # the size of the post layer is inferred from the size of the matrix weight

        syn = self.state.syn
        mem = self.state.mem
        S = self.state.S

        if self.is_recurrent:
            h1 += torch.mm(S, self.weight_rec)

        new_syn = self.alpha * syn + h1
        new_mem = self.beta * mem + new_syn
        #new_mem = torch.clamp(new_mem, min=0)
        #print(torch.mean(new_mem).item())

        mthr = new_mem - self.thr # compare mthr to spiking threshold
        out = activation(mthr)
        rst = out.detach()

        self.state = self.LIFstate(syn=new_syn,
                                   mem=new_mem * (1.0 - rst),
                                   S=out)

        return out


class wLIF(nn.Module):
    LIFstate = namedtuple('LIFstate', ['syn', 'mem', 'S'])

    def __init__(self, nb_inputs, nb_outputs, alpha, beta, train_tau=False, fwd_weight_scale=1.0, mean_fwd_weight=0.0,
                 rec_weight_scale=1.0, layer=None, thr=1, vmin=-10):
        super(wLIF, self).__init__()

        weights = torch.hstack((torch.eye(nb_inputs), torch.zeros((nb_inputs, nb_outputs))))
        self.weight = torch.nn.Parameter(weights, requires_grad=False)

        # self.offset = torch.nn.Parameter(torch.empty((nb_outputs,)), requires_grad=is_trained) # Setting biases.
        # torch.nn.init.normal_(self.offset, mean=0, std=1 / np.sqrt(nb_outputs))

        self.weight_rec = torch.nn.Parameter(torch.empty((nb_outputs, nb_outputs)), requires_grad=True)
        torch.nn.init.normal_(self.weight_rec, mean=0.0, std=rec_weight_scale / np.sqrt(nb_inputs))

        self.alpha = alpha
        self.beta = beta
        self.state = None
        self.thr = thr
        self.vmin = vmin
        self.layer = layer

        if train_tau:
            self.alpha = torch.nn.Parameter(self.alpha, requires_grad=True)
            self.beta = torch.nn.Parameter(self.beta, requires_grad=True)

    def initialize(self, input):
        self.state = self.LIFstate(syn=torch.zeros_like(input, device=input.device),
                                   mem=torch.zeros_like(input, device=input.device),
                                   S=torch.zeros_like(input, device=input.device))

    def reset(self):
        self.state = None

    def forward(self, input):
        h1 = torch.mm(input, self.weight)
        # bs = input.shape[0]
        # h1 = torch.mm(input, self.weight) + self.offset.repeat(bs, 1) * ADD_OFFSET_LIF

        if self.state is None:
            self.initialize(h1) # the size of the post layer is inferred from the size of the matrix weight

        syn = self.state.syn
        mem = self.state.mem
        S = self.state.S

        h1 += torch.mm(S, self.weight_rec)

        new_syn = self.alpha * syn + h1
        new_mem = self.beta * mem + new_syn
        # new_mem = torch.clamp(new_mem, min=0)
        # print(torch.mean(new_mem).item())

        mthr = new_mem - self.thr # compare mthr to spiking threshold
        out = activation(mthr)
        rst = out.detach()

        self.state = self.LIFstate(syn=new_syn,
                                   mem=new_mem * (1.0 - rst),
                                   S=out)

        return out


def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    """ Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

    Args:
    x -- The "current" values

    Keyword args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value
    tmax -- The maximum time returned
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Time to first spike for each "current" x
    """
    idx = x < thr
    x = np.clip(x, thr + epsilon, 1e9)
    T = tau * np.log(x / (x - thr))
    T[idx] = tmax
    return T


def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, shuffle=True):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors.

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    labels_ = np.array(y, dtype=np.int)
    number_of_batches = len(X) // batch_size
    sample_index = np.arange(len(X))

    # compute discrete firing times
    tau_eff = 20e-3 / time_step
    firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=np.int)
    unit_numbers = np.arange(nb_units)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            c = firing_times[idx] < nb_steps
            times, units = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index], device=device)

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1


def vr_responses(X, pos_vr):
    Xvr = np.zeros((X.shape[0], pos_vr.shape[0]))
    for i, pos in enumerate(pos_vr):
        Xvr[:, i] = np.linalg.norm(X - pos, axis=1)
    Xvr = 1 - (Xvr - np.min(Xvr))/(np.max(Xvr) - np.min(Xvr))
    return Xvr


def svm_classification(X, Y, split=0.2, rand=True):
    ntrain = int(X.shape[0] * (1 - split) / 3)
    classes = np.unique(Y)

    if rand:
        p = np.random.permutation(X.shape[0])
        X = X[p]
        Y = Y[p]

    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []

    for i in classes:
        Xtrain.append(X[np.where(Y == i)][:ntrain])
        Ytrain.append(Y[np.where(Y == i)][:ntrain])
        Xtest.append(X[np.where(Y == i)][ntrain:])
        Ytest.append(Y[np.where(Y == i)][ntrain:])

    Xtrain = np.vstack(Xtrain)
    Ytrain = np.hstack(Ytrain)
    Xtest = np.vstack(Xtest)
    Ytest = np.hstack(Ytest)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(Xtrain, Ytrain)

    predtrain = classifier.predict(Xtrain)
    predtest = classifier.predict(Xtest)

    acctrain = metrics.accuracy_score(Ytrain, predtrain)
    acctest = metrics.accuracy_score(Ytest, predtest)

    return acctrain, acctest


def plot_whitevsnowhite(acctrain, acctest, acctrainw, acctestw, step=1):

    maxvr = acctrain.shape[0]
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 4))

    ax[0].title.set_text('Train')
    ax[0].plot(np.arange(1, maxvr + 1, step), np.mean(acctrain, axis=1), label='no-white')
    ax[0].fill_between(np.arange(1, maxvr + 1, step),
                       np.mean(acctrain, axis=1) - np.std(acctrain, axis=1),
                       np.mean(acctrain, axis=1) + np.std(acctrain, axis=1), alpha=0.2)

    ax[0].plot(np.arange(1, maxvr + 1, step), np.mean(acctrainw, axis=1), label='white')
    ax[0].fill_between(np.arange(1, maxvr + 1, step),
                       np.mean(acctrainw, axis=1) - np.std(acctrainw, axis=1),
                       np.mean(acctrainw, axis=1) + np.std(acctrainw, axis=1), alpha=0.2)

  #  ax[0].plot(np.arange(1, maxvr + 1), np.mean(accrawtrain) * np.ones(maxvr), linestyle='dashed', color='grey')

    ax[1].title.set_text('Test')
    ax[1].plot(np.arange(1, maxvr + 1, step), np.mean(acctest, axis=1), label='no-white')
    ax[1].fill_between(np.arange(1, maxvr + 1, step),
                       np.mean(acctest, axis=1) - np.std(acctest, axis=1),
                       np.mean(acctest, axis=1) + np.std(acctest, axis=1), alpha=0.2)

    ax[1].plot(np.arange(1, maxvr + 1, step), np.mean(acctestw, axis=1), label='white')
    ax[1].fill_between(np.arange(1, maxvr + 1, step),
                       np.mean(acctestw, axis=1) - np.std(acctestw, axis=1),
                       np.mean(acctestw, axis=1) + np.std(acctestw, axis=1), alpha=0.2)

    #ax[1].plot(np.arange(1, maxvr + 1), np.mean(accrawtest) * np.ones(maxvr), linestyle='dashed', color='grey')

 #   plt.xticks([1, 5, 10, 15, 20])
    plt.ylim((0.8, 1))

    plt.legend()

    return fig


def whiten(X):
    Xwhite = X - np.mean(X, axis=0)
    pca = PCA()
    pca.fit(Xwhite)
    U = np.matmul(np.linalg.inv((pca.explained_variance_*np.identity(X.shape[1]))**(1/2)),
                  pca.components_)
    return np.matmul(U, Xwhite.transpose()).transpose()


def normalize_std(X):
    Y = np.zeros_like(X)
    std = np.std(X, axis=0)
    for i, val in enumerate(std):
        if val != 0:
            Y[:, i] = X[:, i] / val

    return Y - np.mean(Y, axis=0)


def normalize(X):
    Y = X - np.mean(X)
    return Y / np.mean(np.std(Y, axis=0))


def softWTA(X, beta=0.1):
    W = np.eye(X.shape[1]) - beta*(np.ones((X.shape[1], X.shape[1])) - np.eye(X.shape[1]))
    return np.matmul(W, X.transpose()).transpose()


def prepare_output_data(args):
    """
    Prepare output file which stores parameters and results.
    Return dictionary with hyperparameters.
    """

    out_dict = {'parameters': {}}

    for item in args._get_kwargs():
        out_dict['parameters'][item[0]] = item[1]

    return out_dict


def poisson_spikes(freqs, n_steps, rate_norm=0.1):
    """

    :param freqs: output frequencies (normalized between 0 and 1)
    :param n_steps: number of time bins
    :return: spikes Tensor (n_steps x n_neurons)
    """
    n_inputs = freqs.shape[0]
    return torch.where(torch.rand((n_steps, n_inputs)) < torch.tensor(freqs)*rate_norm, 1., 0.)


def adap_whitening(X):
    X = torch.Tensor(X)

    N = X.size()[1]
    K = int(N * (N + 1) / 2)

    a = 0.001
    b = 0.001

    W = torch.rand(N, K)
    W = W / torch.norm(W, dim=0)

    g = torch.zeros(K)
    g_stack = []
    Y = []
    Z = []

    for i, x in enumerate(X):
        #     y = torch.zeros(N)
        y = torch.linalg.inv(torch.eye(N) + W @ torch.diag(g) @ W.T) @ x
        z = W.T @ y
        g += b * (z * z - torch.ones(K))
        #     print(g)
        g_stack.append(g.clone())
        Y.append(y.clone())
        Z.append(z.clone())

    g_stack = torch.vstack(g_stack)
    Y = torch.vstack(Y)
    Z = torch.vstack(Z)

    return Y.numpy(), Z.numpy(), g_stack.numpy()


def offline_adap_whitening(cov, steps=100000, lr_g=0.001, lr_w=0.00001, alpha=1., K_constrained=False):
    N = cov.size()[0]
    if K_constrained:
        K = N
    else:
        K = int(N * (N + 1) / 2)

    W = torch.rand(N, K)
    W = W / torch.norm(W, dim=0)
    g = torch.zeros(K)
    for i in range(steps):
        M = alpha * torch.eye(N) + W @ torch.diag_embed(g) @ W.T
        grad_M = -pinv(M) @ cov @ pinv(M) + torch.eye(N)
        g -= lr_g * (torch.diag(W.T @ grad_M @ W))
        W -= lr_w * (grad_M @ W @ torch.diag_embed(g))
    return g, W


def adap_whitening_2(X, lr_g=0.001, lr_w=0.00001, alpha=1., K_constrained=False):
    X = torch.Tensor(X)

    N = X.size()[1]

    if K_constrained:
        K = N
    else:
        K = int(N * (N + 1) / 2)

    W = torch.rand(N, K)
    W = W / torch.norm(W, dim=0)

    g = torch.zeros(K)
    g_stack = []
    W_stack = []
    Y = []
    Z = []

    for i, x in enumerate(X):
        #     y = torch.zeros(N)
        y = inv(alpha * torch.eye(N) + W @ torch.diag(g) @ W.T) @ x
        z = W.T @ y
        g += lr_g * (z * z - torch.diag(W.T @ W))
        #     print(g)
        W += lr_w * (y.reshape(-1, 1) @ (g * z).reshape(1, -1) - W @ torch.diag(g))
        g_stack.append(g.clone())
        W_stack.append(W.clone())
        Y.append(y.clone())
        Z.append(z.clone())

    g_stack = torch.vstack(g_stack)
    W_stack = torch.stack(W_stack)
    Y = torch.vstack(Y)
    Z = torch.vstack(Z)

    return Y, Z, g_stack, W_stack
