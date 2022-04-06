"""
Project: multichannel predictions of neural activity.
Author: Kylie Hoyt
Documentation support: Derek Chang
Last updated: 08/29/2022
"""
################### Import reauired packages ###################

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
from numpy import matlib


####################### Data and Globals #######################

fs = 3051.76    # Hz
dt = 1 / fs     # s
spikes = np.load('spike_trains.npy', allow_pickle=True)
spikes = spikes.tolist()

sorted_channels = [5, 14, 22, 30, 39, 55, 58, 77, 90, 1, 3, 4, 17, 18, 20, 47,
                   94, 95, 2, 23, 65, 67, 68, 81, 82, 83, 97, 98, 99, 100, 101,
                   102, 103, 104, 105, 106, 108, 109, 110, 111, 113, 115, 116,
                   117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 136,
                   137, 138, 139, 141, 142, 144, 148, 152, 53, 154, 155, 156, 158,
                    9, 10, 25, 26, 33, 35, 36, 37, 38, 40, 46, 48, 49, 51, 52,
                   53, 54, 56, 62, 70, 72, 73, 75, 80, 84, 85, 86, 88, 89, 96,
                   69, 74, 91, 6, 19, 42, 43, 66, 78, 7, 8, 15, 16, 21, 24, 31,
                   32, 44, 45, 59, 60, 61, 132, 133, 135, 146, 149, 151, 147,
                   150, 160, 134, 130, 34, 50, 71, 6, 79, 87, 92, 93, 107, 123,
                   11, 12, 13, 27, 28, 29, 41, 57, 63, 64, 112, 114, 140, 143, 157, 159]

# 28 most active and representative channel numbers (not indices)
consistent_channels = [4, 45, 46, 59, 70, 71, 72, 73, 75, 76, 78, 82, 83, 84,
                       85, 86, 91, 94, 105, 107, 111, 114, 118,
                       123, 147, 149, 154, 160]

consistent_regions = [0, 1, 3, 1, 3, 2, 3, 3, 3, 2, 4, 5, 5, 3,
                      3, 3, 3, 6, 5, 2, 5, 7, 5,
                      2, 8, 9, 5, 8]
# consistent channels minus least active channels of well-represented regions (22 channels)
short_consistent_channels = [4, 45, 46, 70, 72, 73, 75, 78, 82, 84,
                             86, 91, 94, 105, 107, 114, 118,
                             123, 147, 149, 154, 160]

short_consistent_regions = [0, 1, 3, 3, 3, 3, 3, 4, 5, 3,
                            3, 3, 6, 5, 2, 7, 5,
                            2, 8, 9, 5, 8]

################### Supporting Functions ###################


def plot_spikes_with_prediction(sptrain, predicted_sptrain, dt, nt=2000, t0=0, tit=None, **kws):
    """Plot actual and predicted spike counts.
    Inputs:
      sptrain (1D array): Vector of actual spike counts
      predicted_sptrain (1D array): Vector of predicted spike counts
      dt (number): Duration of each time bin
      nt (number): Number of time bins to plot
      t0 (number): Index of first time bin to plot
      tit (string): Title of plot
      kws: Pass additional keyword arguments to plot()
    Outputs:
      plot
    """
    t = np.arange(t0, t0 + nt) * dt
    f, ax = plt.subplots()
    lines = ax.stem(t, sptrain[:nt], use_line_collection=True)          # Actual Spike Counts
    plt.setp(lines, color=".5")
    lines[-1].set_zorder(1)
    kws.setdefault("linewidth", 3)
    yhat, = ax.plot(t, predicted_sptrain[:nt], **kws)                   # Predicted Spike Counts
    ax.set(
        xlabel="Time (s)",
        ylabel="Spikes",
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend([lines[0], yhat], ["Spikes", "Predicted"])
    plt.title(tit)
    plt.show()


def plot_spike_filter(theta, dt, **kws):
    """Creates plot of estimated weights for spike history model.
    Inputs:
      theta (1D array): Filter weights, not including bias offset
      dt (number): Duration of each time bin
      kws: Pass additional keyword arguments to plot()
    Outputs:
      None
    """
    d = len(theta)
    t = np.arange(-d, 0) * dt
    ax = plt.gca()
    ax.plot(t, theta, marker="o", **kws)
    ax.axhline(0, color=".2", linestyle="--", zorder=1)
    ax.set(
      xlabel="Time before spike (s)",
      ylabel="Filter weight",
    )


def bin_data(data, bin_size=25):
    """Bins data into groups of specified bin_size.
    Inputs:
      data (array): Data to be binned; may be 1D or 2D (sptrain or sptrial)
      bin_size (number): Number of desired data points in each time bin
    Outputs:
      binned_data (array): Data with new shape of (r, c/bin_size)
    """
    sh = np.shape(data)
    if len(sh) == 2:                                # for a 2D data set
        rows = sh[0]
        cols = sh[1]
        num_bins = int(np.floor(cols / bin_size))
        binned_data = np.zeros((rows, num_bins))
        for m in range(rows):
            for n in range(num_bins):
                binned_data[m, n] = np.sum(data[m, n*bin_size:n*bin_size+bin_size-1])
    else:                                           # for a 1D data set
        cols = sh[0]
        num_bins = int(np.floor(cols / bin_size))
        binned_data = np.zeros(num_bins)
        for n in range(num_bins):
            binned_data[n] = np.sum(data[n*bin_size:n*bin_size+bin_size-1])
    return binned_data


def mean_ISI(sptrain):
    """Calculates the mean inter-spike interval of an unbinned spike train.
    Inputs:
      sptrain (1D array): Vector of actual spike counts for a channel
    Outputs:
      ISI.mean() (number): Mean inter-spike interval of a spike train
    """
    c = len(sptrain)
    time = [i * dt for i in range(0, c)]
    spike_times = np.multiply(sptrain, time)                  # time points of sptrain
    non_zeros = np.extract(spike_times > 0, spike_times)      # time points of spikes
    ISI = non_zeros.copy()
    ISI[1:] = np.diff(non_zeros)                              # differences in time between consecutive sptrain
    return ISI.mean()


def clean_spikes(sptrial, thresh, mode="cons"):
    """(mode = "thresh") Removes the channels of a trial that have total spike counts below a specified threshold
    or (mode = "cons") removes the channels of a trial that are not in the list of consistent_channels
    Inputs:
      sptrial (2D array) - All the channels of a trial
      thresh (number) - Only channels with total spike counts > thresh will be kept
      mode (string) - "cons" selects consistent channels mode; "thresh" selects spike count threshold mode
    Outputs:
      cleanspikes[1:, :] (2D array) - The channels of a trial that meet the specified criteria
      channels (1D array) - The channel indices or numbers of the channels that meet the criteria
    """
    cleanspikes = np.ones_like(sptrial[0, :])
    if mode == "thresh":
        channels = []
        for ch in range(len(sptrial)):
            if sum(sptrial[ch, :]) > thresh:                                # extract the channels with > thresh spikes
                cleanspikes = np.row_stack((cleanspikes, sptrial[ch, :]))
                channels.append(ch)
        return cleanspikes[1:, :], channels                                 # channels = channel indices
    elif mode == "cons":
        channels = consistent_channels                                      # select list of desired channel numbers
        for ch in channels:
            cleanspikes = np.row_stack((cleanspikes, sptrial[ch-1, :]))     # extract channels on the list
        return cleanspikes[1:, :], channels                                 # channels = channel numbers


def hist_plot(sptrial, tr=1):
    """Plots a histogram of channel spike counts of a trial with the mean and median.
    Inputs:
      sptrial (2D array): Spike trains of all channels of a trial
      tr (number): Trial being plotted; for plot title
    Outputs:
      plot
    """
    nspikes = np.sum(sptrial, 1)
    mean_spikes = np.mean(nspikes)
    median_spikes = np.median(nspikes)
    plt.hist(nspikes, align='left')
    plt.axvline(mean_spikes, color='red', label='Mean')
    plt.axvline(median_spikes, color='green', label='Median')
    plt.title(f'Number of Spikes per Channel -- Trial {tr}')
    plt.xlabel('Number of Spikes')
    plt.ylabel(f'Number of Channels, Total = {len(nspikes)}')
    plt.legend()
    plt.show()


def make_design_matrix(sptrain):
    """Creates a t-1 time-lag design matrix from a spike train of width 1.
    Inputs:
      sptrain (1D array): spike train of a channel with length T
    Outputs:
      X (2D array): GLM design matrix with shape (T, 1)
    """
    padded_x = np.concatenate([np.zeros(1), sptrain])                        # Add a zero to front of sptrain
    T = len(sptrain)                                               # Total number of time points
    X = np.zeros((T, 1))
    for t in range(T):
        X[t] = padded_x[t:t+1]                                     # Each row t is the 1 points before t in sptrain
    return X


def bin_size_comparison(sptrial, test_channel, bin_times):
    """Creates a plot of the NLLs of a given test_channel and trial using specified bin times
    Inputs:
      sptrial (2D array): All the channels of a trial
      test_channel (number): The test channel to be evaluated
      bin_times (1D array): A list of the times in ms that the spikes are to be binned by
    Outputs:
      plot
    """
    NLL = np.zeros(len(bin_times))
    for bs in range(len(bin_times)):
        NLL[bs] = multi_channel_lnp(sptrial, test_channel, bin_times[bs])
    plt.plot(bin_times, NLL)
    plt.xlabel("Bin Time [ms]")
    plt.ylabel("NLL")
    plt.title(f'Channel {test_channel+1} -- Bin Time Comparison')
    plt.show()


################# Multi Channel Poisson GLM ####################
def multi_neg_log_lik(b_theta, X, y):
    """Calculates the combined negative log likelihood for multiple test channels.
    Inputs:
      b_theta (1D array) - Flattened version of bias offset (b) and filter weights (theta) for each test channel/input channel
      X (2D array) - Design matrix containing spike histories of each input channel
      y (2D array) - Actual spike trains of each test channel
    Outputs:
      -sum_log_lik (number): sum of the negative log likelihoods for each test channel
    """
    sumlin = np.zeros_like(y)
    sum_log_lik = 0
    num_test_ch = len(y)
    b_theta = np.reshape(b_theta, (num_test_ch, -1))
    b_theta = b_theta.T
    for ch in range(num_test_ch):                               # for each test channel
        b = b_theta[-1, ch]
        theta = b_theta[:-1, ch]
        sumlin[ch] = np.matmul(X, theta)                        # (T, #inputchannels) @ (#inputchannels,) -> (T,)
        yhat = np.exp(sumlin[ch]+b)                             # yhat for one test channel at a time
        log_lik = y[ch, :] @ np.log(yhat) - yhat.sum()
        sum_log_lik = sum_log_lik + log_lik
    return -sum_log_lik


def multi_channel_lnp(sptrial, test_channels, bin_time=None, mode="con"):
    """Calculates the mean inter-spike interval of an unbinned spike train.
    Inputs:
      sptrial (2D array) - Spike trains for all channels for a trial
      test_channels (1D array) - List of the channel indices to predict
      bin_time (number) - The time in milliseconds to bin the spike trains by
      mode (string) - "con" selects connectivity matrix, "NLLs" selects heat map, "plot" selects plot predictions
    Outputs:
      theta (3D array) - For mode = "con", filter weights of each input channel on each test channel
      NLLs (1D array) - For mode = "NLLs", negative log likelihood for each test channel
      plot - For mode = "plot"
    """
    #------------------------------                                          Bin and Clean Data
    y = sptrial[test_channels, :]  # actual spike train of test_channel
    if bin_time is not None:  # Bin data if desired, bin time is in milliseconds
        bin_size = int(np.floor(bin_time / 1000 * fs))
        sptrial = bin_data(sptrial, bin_size)
        y = bin_data(y, bin_size)
        dt = bin_time/1000.0
    sptrial, ch = clean_spikes(sptrial, 20, "cons")                         # retrieve the spike trains of the consistent channels
    #------------------------------                                         X and Initial Thetas/b
    X = np.row_stack((np.zeros(len(ch)), sptrial.T))                        # X = spike histories of each input channel
    X = X[:-1, :]                                                           # X.shape = (T, #inputchannels)
    x0 = np.random.normal(0, 0.2, size=(len(ch)))                           # x0 = initial theta of each input channel
    b_theta = np.array(np.concatenate([x0, [-4]]))                          # stack initial b under the theta matrix
    b_theta = np.matlib.repmat(b_theta, len(test_channels), 1)
    #------------------------------                                         Minimize Combined Negative Log Likelihood
    res = minimize(multi_neg_log_lik, b_theta, args=(X, y))                 # Find best thetas and b
    b_theta_LNP = res["x"]
    neg_log_lik = res["fun"]
    #------------------------------                                         Retrieve Optimized Theta's and b's
    b_theta_LNP = np.reshape(b_theta_LNP, (len(test_channels), -1))
    b_theta_LNP = b_theta_LNP.T
    b = b_theta_LNP[-1, :]
    theta = b_theta_LNP[:-1, :]
    if mode == "con":                                                       ############ Connectivity Matrix Mode
        return theta.T
    elif mode == "NLLs":                                                    ############ Heat Map Mode
        sumlin = np.zeros_like(y)
        NLLs = np.zeros(len(test_channels))
        for ch in range(len(test_channels)):
            sumlin[ch] = np.matmul(X, theta[:, ch])
            yhat = np.exp(sumlin[ch]+b[ch])
            log_lik = y[ch] @ np.log(yhat) - yhat.sum()                     # Compute minimized NLL for each channel
            NLLs[ch] = -log_lik
        print(f'Combined NLL = {neg_log_lik:.1f}')
        return NLLs
    elif mode == "plot":                                                    ############ Plot Predictions Mode
        sumlin = np.zeros_like(y)
        for ch in range(len(test_channels)):
            sumlin[ch] = np.matmul(X, theta[:, ch])
            yhat = np.exp(sumlin[ch] + b[ch])
            log_lik = y[ch] @ np.log(yhat) - yhat.sum()
            plot_spikes_with_prediction(y[ch], yhat, dt, nt=len(y[ch]),
                                    tit=f'Multi LNP Fit to Channel {test_channels[ch]+1} -- NLL = {-log_lik:.1f}')
    print(f'Combined NLL = {neg_log_lik:.1f}')
    return


tr = 3                                                  # trial to be modeled
test_ch = [3, 44, 45]                                   # channel indices to predict
multi_channel_lnp(spikes[tr], test_ch, 40, "plot")


################# Heat Map ####################
def heat_map(trials=None, test_channels=None, bin_time=None):
    """Plots a heat map of the negative log likelihoods for specified test channels and trials.
    Inputs:
      trials (1D array) - List of trials to predict
      test_channels (1D array) - List of the channel indices to predict
      bin_time (number) - The time in milliseconds to bin the spike trains by
    Outputs:
      plot
    """
    if trials is None:
        trials = range(1, 6)
    if test_channels is None:
        test_channels = [x-1 for x in consistent_channels]                                      # indices of consistent channels
    NLLs = np.zeros((len(test_channels), len(trials)))                                          # NLLs.shape = (#testchannels, #testtrials)
    for t in range(len(trials)):                                                                # for each trial
        NLLs[:, t] = multi_channel_lnp(spikes[trials[t]], test_channels, bin_time, "NLLs")      # determine the combined NLLs of each test channel
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    # Sort NLLs by region
    NLLs = np.column_stack((consistent_regions, consistent_channels, NLLs))                     # label NLLs with channel numbers and region codes
    NLLs = NLLs[NLLs[:, 0].argsort()]                                                           # sort channels by region
    channellabels = [int(x) for x in NLLs[:, 1]]
    NLLs = NLLs[:, 2:]
    # print(NLLs)  # To store NLLs for later use
    ax = sns.heatmap(NLLs, linewidths=0.5, cmap="YlGnBu",
                     ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"},
                     xticklabels=trials, yticklabels=channellabels)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Channel")
    ax.set_title("Negative Log Likelihood")
    plt.show()
    return


# heat_map(bin_time=40)                                         # un-comment to run heat_map

# Manually Generated Heat Map Using Short_Consistent_Channels
NLL2_3_4 = [[4.36353402e+01, 4.21153882e+01, 4.39605654e+01], [2.81354262e+01, 3.29367114e+01, 3.86337589e+01],
            [4.61631566e+01, 4.88631481e+01, 3.95994935e+01], [3.25290432e+01, 2.99047313e+01, 4.64642728e+01],
            [1.26137056e+01, 9.22741131e+00, 2.68995410e+01], [4.98580868e+01, 4.87792061e+01, 5.54317818e+01],
            [4.36343884e+01, 4.07171245e+01, 5.02427780e+01], [5.05927756e+01, 4.39377063e+01, 5.27614225e+01],
            [3.42329282e+01, 3.79801536e+01, 4.04835768e+01], [2.80127253e+01, 2.85865464e+01, 5.02468067e+01],
            [4.84837444e+01, 4.16346046e+01, 4.54458456e+01], [2.76453770e+01, 2.88754493e+01, 3.46852703e+01],
            [3.86824006e+01, 3.48797361e+01, 4.72401150e+01], [1.95095275e+01, 9.00000000e+00, 4.06553846e+01],
            [4.84602948e+01, 4.36893880e+01, 4.31989538e+01], [2.46354802e+01, 2.67538792e+01, 3.32878700e+01],
            [4.00000000e+00, 4.00000000e+00, 6.00000107e+00], [2.27823437e+01, 7.61370564e+00, 9.61373424e+00],
            [2.37562766e+01, 7.61379054e+00, 1.86805706e+01], [3.30456551e+01, 2.74538697e+01, 2.72989901e+01],
            [2.17153491e+01, 7.22741128e+00, 5.40832647e+00], [1.32275612e+01, 2.57272335e+01, 2.54989095e+01]]
NLL5_7 = [[3.45936042e+001, 4.53781862e+001], [4.28776874e+001, 3.55066144e+001],
          [4.88031316e+001, 4.73106802e+001], [2.86402587e+001, 2.84435444e+001],
          [8.00000006e+000, 1.00000780e+001], [3.76910739e+001, 4.59312407e+001],
          [3.96037904e+001, 4.85683554e+001], [4.47274853e+001, 4.94945806e+001],
          [9.31808356e+000, 3.12630858e+001], [3.57054131e+001, 3.91428026e+001],
          [4.40694898e+001, 4.06445814e+001], [3.46308361e+001, 3.16892127e+001],
          [3.58800534e+001, 4.37357839e+001], [2.47367321e+001, 2.83895041e+001],
          [3.06587351e+001, 3.84132612e+001], [1.20000000e+001, 3.60970203e+001],
          [2.82331498e+001, 3.00860812e+001], [7.00000000e+000, 1.86759530e+001],
          [2.00000000e+000, 6.00000000e+000], [2.71047734e+001, 2.25426064e+001],
          [3.98511216e+001, 5.61370564e+000], [7.61370564e+000, 4.22741128e+000]]
NLL9_12_13 = [[2.88876948e+01, 2.80822934e+01, 2.15255166e+01], [4.62037413e+01, 4.37519868e+01, 2.75116767e+01],
              [5.08132297e+01, 4.23394504e+01, 4.32254700e+01], [3.11772580e+01, 1.87784395e+01, 3.27188571e+01],
              [8.00000000e+00, 8.00000001e+00, 1.00000000e+00], [4.44974718e+01, 4.45160316e+01, 3.53699073e+01],
              [3.80940539e+01, 4.47722451e+01, 4.59283415e+01], [4.03633063e+01, 4.86600882e+01, 3.72585078e+01],
              [7.61370564e+00, 8.31795092e+00, 3.63167373e+01], [4.59686954e+01, 3.09519945e+01, 2.65851721e+01],
              [3.96354621e+01, 4.18239863e+01, 3.94778842e+01], [3.39658465e+01, 3.30392242e+01, 3.43621497e+01],
              [3.65770845e+01, 3.75996247e+01, 3.26238750e+01], [3.03292199e+01, 2.15588800e+01, 2.32289295e+01],
              [3.49121091e+01, 2.83286758e+01, 3.89300921e+01], [2.72944198e+01, 1.81229688e+01, 1.03178688e+01],
              [2.86491567e+01, 1.78545223e+01, 8.00002949e+00], [8.00011082e+00, 6.00000003e+00, 6.00000000e+00],
              [3.61370564e+00, 4.61370565e+00, 3.37627198e+01], [2.18815662e+01, 2.66654061e+01, 6.61370564e+00],
              [1.59010256e+01, 1.62326963e+01, 9.02203191e+00], [9.00000000e+00, 4.61370579e+00, 2.69407430e+01]]

NLL_collection = np.column_stack((NLL2_3_4, NLL5_7, NLL9_12_13))
grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
trs = [2, 3, 4, 5, 7, 9, 12, 13]
chlabels = [int(x) for x in short_consistent_channels]
ax = sns.heatmap(NLL_collection, linewidths=0.5, cmap="YlGnBu", ax=ax, cbar_ax=cbar_ax,
                 cbar_kws={"orientation": "horizontal"}, xticklabels=trs, yticklabels=chlabels)
ax.set_xlabel("Trial")
ax.set_ylabel("Channel")
ax.set_title("Negative Log Likelihood")
plt.show()


################# Representative Channels ####################
def active_channel_sort():
    """Ranks the channels by percentage of trials with spikes and average spikes per trial.
    Inputs:
      None
    Outputs:
      summed_rank (2D array) - [channel label, summed rank, percentage of trials with spikes, avg spikes per trial]
      for each channel
    """
    #                                                                 -- Count Spikes for Each Channel in Each Trial
    counts = np.zeros((len(spikes[1]), len(spikes)))
    for t in range(1, len(spikes)+1):
        counts[:, t-1] = np.sum(spikes[t], axis=1)
    #                                                                 -- Average Spikes Per Trial
    avg_spikes = np.divide(np.sum(counts, axis=1), len(spikes))          # spikes for each channel for all trials/201
    avg_spikes = np.column_stack((range(1, 161), avg_spikes))            # label channels
    avg_spikes = avg_spikes[avg_spikes[:, 1].argsort()]                  # sort by average
    avg_spikes = np.column_stack((range(0, 160), avg_spikes))            # score for average
    avg_spikes = avg_spikes[avg_spikes[:, 1].argsort()]                  # sort by channel label
    #                                                                 -- Percentage of Trials with Spikes
    perc_spikes = np.where(counts >= 1, 1, 0)                            # boolean for spikes for each channel and trial
    perc_spikes = np.divide(np.sum(perc_spikes, axis=1), len(spikes))    # fraction of trials with spikes for each ch
    perc_spikes = np.multiply(perc_spikes, 100)                          # percent
    perc_spikes = np.column_stack((range(1, 161), perc_spikes))          # label channels
    perc_spikes = perc_spikes[perc_spikes[:, 1].argsort()]               # sort by percentage
    perc_spikes = np.column_stack((range(0,160), perc_spikes))           # score for percentage
    perc_spikes = perc_spikes[perc_spikes[:, 1].argsort()]               # sort by channel label
    #                                                                 -- Summed Rank
    summed_rank = np.zeros((160, 3))
    summed_rank[:, 0] = perc_spikes[:, 0] + avg_spikes[:, 0]             # column of summed rank
    summed_rank[:, 1] = perc_spikes[:, 2]                                # column of percentage of trials with spikes
    summed_rank[:, 2] = avg_spikes[:, 2]                                 # column of average spikes per trial
    summed_rank = np.column_stack((range(1, 161), summed_rank))          # label channels
    summed_rank = summed_rank[summed_rank[:, 1].argsort()]               # sort by summed rank
    np.set_printoptions(suppress=True)                                   # not scientific notation
    print(summed_rank)


################# Connectivity Matrix ####################
def Connectivity_Matrix(trial):
    ch = [x-1 for x in consistent_channels]                                     # test channels are the consistent channels
    theta = multi_channel_lnp(spikes[trial], ch, 40, "con")                     # retrieve optimized thetas as (test channels, input channels)
    sq = np.column_stack((consistent_regions, consistent_channels, theta))
    sq = sq[sq[:, 0].argsort()]                                                 # sort rows by region
    channellabels = [int(x) for x in sq[:, 1]]
    sq = sq[:, 2:]
    sq = np.row_stack((sq, consistent_channels, consistent_regions))
    sq = sq[sq[-1, :].argsort()]                                                # sort columns by region
    sq = sq[:-2, :]
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    ax = sns.heatmap(sq, linewidths=0.5,
                     ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"},
                     xticklabels=channellabels, yticklabels=channellabels)
    ax.set_xlabel("Input Channel")
    ax.set_ylabel("Test Channel")
    ax.set_title(f'Connectivity Matrix - Trial {trial}')
    plt.show()
    return


# Connectivity_Matrix(1)                                              # un-comment to run Connectivity_Matrix
