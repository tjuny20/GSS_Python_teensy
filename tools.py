import numpy as np
import torch
import torch.nn as nn
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.decomposition import PCA
from torch.linalg import inv, eig, pinv
from scipy.ndimage import gaussian_filter1d
import csv, pickle
from datetime import datetime


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


def load(filename, reduced=True):
    sensor_data = []
    times = []
    responding_sens = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0]
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # times = [row[0] for row in reader]
        for row in reader:
            if row[0] =='Timestamp':
                continue
            else:
                times.append(row[0])
                values = []
                for i in range(17):
                    b1 = int(row[2*i+1])
                    b2 = int(row[2*i+2])
                    values.append(int.from_bytes([b1, b2], byteorder="little"))
                sensor_data.append(values)
    sensor_data = np.array(sensor_data)
    if reduced:
        sensor_data = np.delete(sensor_data, np.where(np.array(responding_sens)==0)[0], axis=1)
    sequence = pickle.load(open('data/1_300_20_sequence.pkl', 'rb'))
    # Convert to seconds
    times_sec = []
    for dt_str in times:
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        times_sec.append(seconds)
    sequence_sec = []
    for dt_str in sequence:
        dt = datetime.strptime(dt_str[0], '%a %b %d %H:%M:%S %Y')
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        sequence_sec.append(seconds)
    times_sec = np.array(times_sec)
    sequence_sec = np.array(sequence_sec)
    return sensor_data, sequence, times_sec, sequence_sec

def load(filename, reduced=True):
    sensor_data = []
    times = []
    responding_sens = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0]
    with open(f'data/{filename}.csv', 'r') as f:
        reader = csv.reader(f)
        # times = [row[0] for row in reader]
        for row in reader:
            if row[0] =='Timestamp':
                continue
            else:
                times.append(row[0])
                values = []
                for i in range(17):
                    b1 = int(row[2*i+1])
                    b2 = int(row[2*i+2])
                    values.append(int.from_bytes([b1, b2], byteorder="little"))
                sensor_data.append(values)
    sensor_data = np.array(sensor_data)
    if reduced:
        sensor_data = np.delete(sensor_data, np.where(np.array(responding_sens)==0)[0], axis=1)
    sequence = pickle.load(open(f'data/{filename}_sequence.pkl', 'rb'))
    # Convert to seconds
    times_sec = []
    for dt_str in times:
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        times_sec.append(seconds)
    sequence_sec = []
    for dt_str in sequence:
        dt = datetime.strptime(dt_str[0], '%a %b %d %H:%M:%S %Y')
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        sequence_sec.append(seconds)
    times_sec = np.array(times_sec)
    sequence_sec = np.array(sequence_sec)
    return sensor_data, sequence, times_sec, sequence_sec

def split(data, delay = 1.5, t_baseline = 300, n_train = 225, n_frames=None):

    sensor_data = data[0]
    times_sec = data[1]
    sequence_sec = data[2]
    baseline = np.mean(sensor_data[:t_baseline], axis=0)    # Add baseline substraction
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    counts = np.zeros((3))

    for i, t in enumerate(sequence_sec):
        try:
            flags = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1] + delay)
        except IndexError:
            flags = (times_sec > sequence_sec[i])
        sample = sensor_data[flags][:18]

        if counts[sequence[i][1]-1] < n_train//3:
            X_train.append(sample.flatten())
            Y_train.append(sequence[i][1]-1)
            counts[sequence[i][1]-1] += 1
        else:
            if n_frames is not None:
                X_test.append(sample[:n_frames].flatten())
            else:
                X_test.append(sample.flatten())
            Y_test.append(sequence[i][1]-1)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    return X_train, Y_train, X_test, Y_test

def estimate_derivative(signal, dt=1):
    """
    Estimate the first derivative of a signal using second-order central difference.

    Parameters:
        signal (np.ndarray): 1D array of signal values.
        dt (float): Sampling interval (time step).

    Returns:
        np.ndarray: Estimated first derivative of the signal.
    """
    n = len(signal)
    derivative = np.zeros_like(signal)

    # Central differences for interior points
    for i in range(1, n - 1):
        derivative[i] = (signal[i + 1] - signal[i - 1]) / (2 * dt)

    # Forward difference for the first point
    derivative[0] = (signal[1] - signal[0]) / dt

    # Backward difference for the last point
    derivative[-1] = (signal[-1] - signal[-2]) / dt

    return derivative

def pseudoderivative(signal, dt=1):
    """
    Estimate the first derivative of a signal using first-order backward difference.

    Parameters:
        signal (np.ndarray): 1D array of signal values.
        dt (float): Sampling interval (time step).

    Returns:
        np.ndarray: Estimated first derivative of the signal.
    """
    n = len(signal)
    derivative = np.zeros_like(signal)

    # Backward difference
    for i in range(1, n):
        derivative[i] = (signal[i] - signal[i-1]) / dt

    # Zero difference for the first point
    derivative[0] = 0.

    return derivative

def find_blocks(arr: np.ndarray, max_len: int = None):
    change_points = np.where(arr[1:] != arr[:-1])[0] + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points - 1, len(arr) - 1]

    blocks = []
    for s, e in zip(starts, ends):
        val = int(arr[s])
        length = e - s + 1
        if max_len is None or length <= max_len:
            blocks.append([int(s), int(e), val])
        else:
            for i in range(s, e + 1, max_len):
                j = min(i + max_len - 1, e)
                blocks.append([int(i), int(j), val])
    return blocks

def add_white_noise_cols(X, snr_db=None, snr_linear=None, seed=None, eps=1e-12):
    """
    Add zero-mean white Gaussian noise to an array of shape (N, 8) so that each
    column attains the specified SNR.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (N, 8). (Works for any (N, M); columns treated independently.)
    snr_db : float, optional
        Desired SNR in dB (per column). Provide exactly one of snr_db or snr_linear.
    snr_linear : float, optional
        Desired SNR as a linear ratio (signal_power / noise_power), per column.
    seed : int, optional
        RNG seed for reproducibility.
    eps : float
        Tiny value to avoid divide-by-zero for silent columns.

    Returns
    -------
    Y : np.ndarray
        Noisy signal, same shape as X.
    noise : np.ndarray
        The noise that was added, same shape as X.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 8:
        raise ValueError("X must be 2D with shape (N, 8).")

    if (snr_db is None) == (snr_linear is None):
        raise ValueError("Provide exactly one of snr_db or snr_linear.")

    if snr_linear is None:
        snr_linear = 10.0 ** (snr_db / 10.0)

    # Signal power per column (mean over time N)
    sig_power = np.mean(X**2, axis=0, keepdims=True)  # shape (1, 8)

    # Target noise power per column
    noise_power = sig_power / (snr_linear + eps)

    # Std per column; broadcasts down rows
    noise_std = np.sqrt(np.maximum(noise_power, 0.0))  # shape (1, 8)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=1.0, size=X.shape) * noise_std

    Y = X + noise
    return Y, noise

def plot_two_intervals(i0, i1, j0, j1,
                       times_sec, sequence_sec, sequence, z_out,
                       max_len=20, sigma=2, savepath='figs/hd_out.pdf'):
    """
    Pretty plotting of z_out activity + block coloring for two index intervals.
    """

    # Global style
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10
    })

    cm = plt.get_cmap('tab10')
    trace_colors = plt.cm.Set2.colors  # softer palette for neurons

    # Build colour array
    colour = np.zeros_like(times_sec)
    for i, t in enumerate(sequence_sec):
        try:
            flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1])
        except IndexError:
            flag = (times_sec > sequence_sec[i])
        colour[flag] = int(sequence[i][1])

    # Convert index pairs (i0,i1), (j0,j1) into time indices
    intervals = []
    for a, b in [(i0, i1), (j0, j1)]:
        t0_sec = sequence_sec[a]
        t1_sec = sequence_sec[b]
        t0 = np.abs(times_sec - t0_sec).argmin()
        t1 = np.abs(times_sec - t1_sec).argmin()
        intervals.append((t0, t1))

    # Make figure with two subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 6),
                           sharey=True,
                           gridspec_kw={'hspace': 0.35})

    for k, (t0, t1) in enumerate(intervals):
        blocks = find_blocks(colour[t0:t1], max_len=max_len)

        # Plot z_out traces
        for i in range(z_out.shape[1]):
            smoothed = gaussian_filter1d(z_out[t0:t1, i], sigma=sigma)
            ax[k].plot(smoothed,
                       label=f'Neuron {i+1}',
                       color=cm(i),
                       lw=1.5, alpha=0.85)

        # Highlight blocks as light shaded regions
        for block in blocks:
            ax[k].axvspan(block[0], block[1],
                          facecolor=cm(block[2]-1), alpha=0.15)

        # Formatting
        ax[k].spines[['top', 'right']].set_visible(False)
        ax[k].set_xlim([0, t1 - t0])
        ax[k].set_ylim([0., 1.])

        # Titles and labels
        titles = ['Train', 'Test']
        ax[k].text(0.99, 1.06, f"{titles[k]}",
           transform=ax[k].transAxes,
           rotation=0, va="center", ha="right",
           fontsize=16,
                   )
        if k == 1:
            ax[k].set_xlabel("# samples")
        elif k==0:
            ax[k].set_xticklabels([])
        ax[k].set_ylabel("Activity (a.u.)")
        ax[k].set_yticks([0,1])

    # Single legend at bottom
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               frameon=False,
               loc="center left",
               bbox_to_anchor=(0.95, 0.83),  # just outside the right side
               ncol=1)

    # plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(savepath, bbox_inches='tight')
    return fig, ax

def train(sensor_data, sequence, times_sec, sequence_sec,
          n_hd=10000, n_out=3, k=10, p=0.5, w_teacher=1.,
          normalized=False, whitened=False, rng_seed=42):

    rng = np.random.default_rng(rng_seed)  # for reproducibility
    if normalized:
        sensor_data_norm = (sensor_data - np.mean(sensor_data, axis=0))/ np.std(sensor_data, axis=0)
    else:
        sensor_data_norm = sensor_data
    if whitened:
        x_dense = adap_whitening_2(sensor_data_norm)
    else:
        x_dense = sensor_data_norm

    n_dense = x_dense.shape[1]

    labels = np.zeros_like(times_sec)
    for i, t in enumerate(sequence_sec):
        try:
            flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1])
        except IndexError:
            flag = (times_sec > sequence_sec[i])
        labels[flag] = int(sequence[i][1])


    W_hd = np.random.binomial(n=1, p=0.05, size=(n_hd, n_dense))  #Test random sparse weights
    x_hd = x_dense @ W_hd.T
    z_hd = np.where(np.argsort(x_hd)<k, 1., 0)
    W_out = np.zeros((n_out, n_hd))
    W = np.zeros((n_out, n_hd))

    for i, row in enumerate(z_hd):
        if labels[i] != 0:
            active_idx = np.flatnonzero(row)
            to_flip = active_idx[rng.random(active_idx.size) < p]  # Bernoulli(p) per active index# indices where z_hd==1
            W_out[int(labels[i]) - 1, to_flip] = 1. / k
        out = row @ W_out.T

    return W_hd, W_out

def test(sensor_data, sequence, times_sec, sequence_sec,
         W_hd, W_out,
         n_hd=10000, n_out=3, k=10, integration_delay=0.,
         normalized=False, whitened=False):

    if normalized:
        sensor_data_norm = (sensor_data - np.mean(sensor_data, axis=0))/ np.std(sensor_data, axis=0)
    else:
        sensor_data_norm = sensor_data
    if whitened:
        x_dense = adap_whitening_2(sensor_data_norm)
    else:
        x_dense = sensor_data_norm

    x_hd = x_dense @ W_hd.T
    z_hd = np.where(np.argsort(x_hd)<k, 1., 0)
    z_out = np.zeros((z_hd.shape[0],  n_out))
    for i, row in enumerate(z_hd):
        out = row @ W_out.T
        z_out[i] = out

    z_pred = np.zeros_like(sequence_sec)
    z_true = np.zeros_like(sequence_sec)
    for i, t in enumerate(sequence_sec):
        try:
            flag = (times_sec > sequence_sec[i] + integration_delay) & (times_sec < sequence_sec[i+1])
        except IndexError:
            flag = (times_sec > sequence_sec[i] + integration_delay)
        z_pred[i] = np.argsort(np.sum(z_out[flag], axis=0))[-1] + 1
        z_true[i] = sequence[i][1]
    test_acc = metrics.accuracy_score(z_true, z_pred)

    return test_acc

def split(sensor_data, sequence, times_sec, sequence_sec, idx_split=450):

    labels = np.zeros_like(times_sec)
    for i, t in enumerate(sequence_sec[:idx_split]):
        try:
            flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1])
        except IndexError:
            flag = (times_sec > sequence_sec[i])
        labels[flag] = int(sequence[i][1])

    idx_last_flag = np.where(labels != 0)[0][-1]

    return sensor_data[:idx_last_flag], sequence[:idx_split], times_sec[:idx_last_flag], sequence_sec[:idx_split], \
           sensor_data[idx_last_flag:], sequence[idx_split:], times_sec[idx_last_flag:], sequence_sec[idx_split:]

def get_samples(sensor_data, sequence, times_sec, sequence_sec,
                idx_split_0=0, idx_split=450, t_training_delay=0., n_frames=None):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    count = 0

    for i, t in enumerate(sequence_sec):
        try:
            flags = (times_sec > sequence_sec[i] + t_training_delay) & (times_sec < sequence_sec[i+1])
        except IndexError:
            flags = (times_sec > sequence_sec[i] + t_training_delay)
        sample = sensor_data[flags][:18]
        t_sample = times_sec[flags]

        if i==387:   # Remove bad samples
            continue

        if idx_split_0 <= count < idx_split:
            X_train.append(sample.flatten())
            Y_train.append(sequence[i][1]-1)
            count += 1
        else:
            if n_frames is not None:
                X_test.append(sample[:n_frames].flatten())
            else:
                X_test.append(sample.flatten())
            Y_test.append(sequence[i][1]-1)


    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test

def find_blocks(labels, max_len=20, ignore=0):
    """
    Split a 1D array of integer labels into contiguous blocks.
    - Blocks with label == `ignore` (default 0) are skipped.
    - Long runs are split into chunks of at most `max_len`.
    Returns: list of (start_idx, end_idx, label) with end_idx exclusive.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return []

    blocks = []
    start = 0
    prev = labels[0]

    # walk + flush on change (and once at the end)
    for i in range(1, n + 1):
        cur = labels[i] if i < n else None
        if cur != prev:
            if prev != ignore:
                run_start, run_end, lab = start, i, int(prev)
                # chunk the run to respect max_len
                s = run_start
                while s < run_end:
                    e = min(s + max_len, run_end)
                    blocks.append((s, e, lab))
                    s = e
            start = i
            prev = cur

    return blocks

