'''
Module for plexon file analysis
'''

from plexon import plexfile
import matplotlib.pyplot as plt
import plotutil
import numpy as np

class PlexonData(object):
    def __init__(self, fname, tslice=slice(None, None)):
        self.plx = plexfile.openFile(fname)
        self.spike_waveforms = self.plx.spikes[tslice].waveforms
        self.spike_timestamps = self.plx.spikes[tslice].data

    def get_wfs(self, chan, unit):
        inds, = np.nonzero((self.spike_timestamps['chan'] == chan) * (self.spike_timestamps['unit'] == unit))
        return self.spike_waveforms[inds]

    def plot_wfs(self, chan, ax_sorted=None, ax_unsorted=None):
        if ax_unsorted == None or ax_sorted == None:
            plt.figure()
            ax_sorted, ax_unsorted = plotutil.subplots(1, 2, hold=True, return_flat=True)

        colors = ['red', 'blue', 'green', 'black', 'magenta']
        for unit in [1, 2, 3, 4]:
            inds, = np.nonzero((self.spike_timestamps['chan'] == chan) * (self.spike_timestamps['unit'] == unit))
            if len(inds > 0):
                unit_wfs = self.spike_waveforms[inds]
                xlist = []
                ylist = []
                d = np.arange(32)
                for wf in unit_wfs:
                    xlist.extend(d)
                    xlist.append(None)
                    ylist.extend(wf)
                    ylist.append(None)

                ax_sorted.plot(xlist, ylist, color=colors[unit])

        for unit in [0]:
            inds, = np.nonzero((self.spike_timestamps['chan'] == chan) * (self.spike_timestamps['unit'] == unit))
            if len(inds) > 0:
                unit_wfs = self.spike_waveforms[inds]
                # ax_unsorted.plot(unit_wfs.T, color=colors[unit])
                xlist = []
                ylist = []
                d = np.arange(32)
                for wf in unit_wfs:
                    xlist.extend(d)
                    xlist.append(None)
                    ylist.extend(wf)
                    ylist.append(None)

                ax_unsorted.plot(xlist, ylist, color=colors[unit])                

        ## Set y limits of plot
        sorted_ylim = ax_sorted.get_ylim()
        unsorted_ylim = ax_unsorted.get_ylim()
        ylim = (min(sorted_ylim[0], unsorted_ylim[0]), max(sorted_ylim[1], unsorted_ylim[1]))
        ax_unsorted.set_ylim(ylim)
        ax_sorted.set_ylim(ylim)

    def plot_all_wfs(self, plot_dir=''):
        # plt.figure()
        # ax_sorted, ax_unsorted = plotutil.subplots(1, 2, hold=True, return_flat=True)
        for k in range(1, 257):
            self.plot_wfs(k)
            plotutil.save(plot_dir, '%d.pdf' % k, dpi=50)
            print k