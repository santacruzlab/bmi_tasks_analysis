#!/usr/bin/python
'''
Test the impact of changing the OFC parameter on the PPF speed
'''
from db import dbfunctions as dbfn
from db.tracker import models
from tasks import performance
import plotutil
import numpy as np
import matplotlib.pyplot as plt

#blocks = [2218, 2248, 2216]
#blocks = [2260, 2261, 2262, 2264, 2265, 2266, 2267]
blocks = [2274, 2275, 2276, 2277, 2281, 2282]
task_entry_set = dbfn.TaskEntrySet(blocks)

bins = np.arange(0.5, 40, 0.5)

labels = []
for te in task_entry_set.task_entries:
    if 'PPF' in te.decoder_type: labels.append(str(te.params['tau']))
    elif 'KF' in te.decoder_type: labels.append('KF')

plt.close('all')
plt.figure(facecolor='w')
axes = plotutil.subplots(2, 1, return_flat=True, hold=True)
#task_entry_set.histogram(lambda te: te.intended_kin_norm(slice(3,6)), axes[0], bins)
task_entry_set.histogram(lambda te: te.cursor_speed(), axes[1], bins, labels=labels)
#task_entry_set.histogram(lambda te: te.cursor_speed('assist_off'), axes[1], bins, labels=labels)
plt.legend()

axes[0].set_xticks([])

plotutil.xlabel(axes[0], 'Est. of intended speed (cm/s)')

plotutil.xlabel(axes[1], 'Actual speed during CLDA (no assist) (cm/s)')
plotutil.set_xlim(axes[0], axis='y')
plotutil.set_xlim(axes[1], axis='y')
plotutil.ylabel(axes, 'Density')
plotutil.set_xlim(axes, [0,40])

plt.savefig('/storage/plots/intended_vs_actual_speed.png', bbox_inches='tight')
plt.show()
