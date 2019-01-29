# coding: utf-8
from db import dbfunctions as dbfn
from db.tracker import models
from tasks import performance
import plotutil
import tables
from scipy.stats import ttest_ind, ttest_rel
import numpy as np
import matplotlib.pyplot as plt
import datetime
from itertools import izip


kf_ppf_pairs = [(2310, 2315), (2325, 2330), (2380, 2378), (2424, 2425), (2446, 2440), (2441, 2445), (2457, 2455), (2465, 2469), (2613, 2609), (2650, 2644), (2661, 2662), (2674, 2670), (2689, (2691,2693)), (2749, 2738), ((2791,2795), 2796), (2841, (2837,2838)), (2857, 2858), (2860, 2859), (2869, (2871,2872,2873)), (2908, 2906)]

kf_lc_pairs = [(2325, 2332), (2424, 2426), (2446, 2442), (2457, 2454), (2465, (2468,2470)), (2613, 2608), (2650, (2646,2647)), (2661, 2657), (2674, 2668), (2689, 2690), (2749, 2740), ((2791,2795), (2798,2806))]

lc_lf_pairs = [(2468, 2467), (2470, 2471), (2495, 2493), (2556, 2555), (2608, 2610), (2619, (2620,2623)), ((2634,2635), 2631), ((2646,2647), 2645), (2657, 2658), (2668, 2669), (2713, (2714,2716)), (2740, 2739), (2759, 2760), ((2782,2783), 2781)]

lf_ppf_pairs = [(2467, 2469), (2493, 2494), (2573, 2574), (2579, 2578), (2610, 2609), ((2620,2623), 2618), (2631, 2632), (2645, 2644), (2663, 2656), (2658, 2662), (2669, 2670), (2739, 2738), (2781, 2784), (2760, 2761), ((2798,2806), 2796), (2820, 2819), (2824, 2826)]

# Ignoring pairs (2310, 2315), (2325, 2330) b/c SSKF modificaiton was not impl
kf_blocks_1  = [2380, 2424, 2441, 2446, 2457, 2465, 2613]
ppf_blocks_1 = [2378, 2425, 2445, 2440, 2455, 2469, 2609]

ppflc_blocks_2 = [2426, 2442, 2454, 2468, 2608]
kf_blocks_2 = [2424, 2441, 2457, 2465, 2613]

ppflf_blocks_3 = [2493, 2467, 2471, 2610]
ppflc_blocks_3 = [2495, 2468, 2470, 2608]

ppf_blocks_4 = [2494, 2469, 2578, 2574, 2609]
ppflf_blocks_4 = [2493, 2467, 2579, 2573, 2610]

all_kf = [2380, 2441, 2446, 2457, 2465, 2424, 2427]
all_ppflc = [2426, 2442, 2454, 2468, 2495, 2470, 2332]
all_ppflf = [2493, 2467, 2471]
all_ppf = [2378, 2425, 2440, 2445, 2455, 2469, 2494, 2330]

kf_block_set_1 = dbfn.TaskEntrySet(kf_blocks_1, name='KF1')
ppf_block_set_1 = dbfn.TaskEntrySet(ppf_blocks_1, name='PPF1')

ppflc_block_set_2 = dbfn.TaskEntrySet(ppflc_blocks_2, name='PPF_LC2')
kf_block_set_2 = dbfn.TaskEntrySet(kf_blocks_2, name='KF2')

ppflf_block_set_3 = dbfn.TaskEntrySet(ppflf_blocks_3, name='PPF_LF3')
ppflc_block_set_3 = dbfn.TaskEntrySet(ppflc_blocks_3, name='PPF_LC3')
ppf_block_set_4 = dbfn.TaskEntrySet(ppf_blocks_4, name='PPF4')
ppflf_block_set_4 = dbfn.TaskEntrySet(ppflf_blocks_4, name='PPF_LF4')

all_kf_blocks = dbfn.TaskEntrySet(all_kf, name='KFALL')
all_ppflc_blocks = dbfn.TaskEntrySet(all_ppflc, name='PPFLCALL')
all_ppflf_blocks = dbfn.TaskEntrySet(all_ppflf, name='PPFLFALL')
all_ppf_blocks = dbfn.TaskEntrySet(all_ppf, name='PPFALL')

class CombinedTaskEntry(dbfn.TaskEntrySet):
    def __init__(self, *args, **kwargs):
        super(CombinedTaskEntry, self).__init__(*args, **kwargs)
        assert len(np.unique(map(task_type, self.task_entries))) == 1

    def __getattr__(self, attr):
        if attr in ['id', 'date', 'decoder_type', 'task_update_rate', 'feedback_rate']:
            return getattr(self.task_entries[0], attr)
        else:
            return np.hstack(self.map(lambda te: getattr(te, attr)))

    def decoders_match(self):
        return len(np.unique([x.decoder.name for x in self.task_entries])) == 1

    @property
    def decoder_name(self):
        if self.decoders_match():
            return self.task_entries[0].decoder.name
        else:
            raise ValueError("Decoders don't have the same name!")

    @property
    def n_trials(self):
        return sum(self.__getattr__('n_trials'))

    @property
    def n_rewards(self):
        return sum(self.__getattr__('n_rewards'))

    @property
    def length(self):
        return sum(map(lambda x: x.length, self.task_entries))

    @property
    def trials_per_min(self):
        return (float(self.n_trials) / self.length) * 60

    @property
    def perc_correct(self):
        n_success = 0
        N = 0
        for te in self.task_entries:
            trial_end_types = te.trial_end_types
            n_success += float(trial_end_types['success'])
            N += (trial_end_types['success'] + trial_end_types['timeout'] + sum(trial_end_types['hold_error'][1:]))
        return n_success / N

    def get_perc_correct(self, n_trials=None):
        return self.perc_correct

    def get_trials_per_min(self, n_trials=None):
        if self.n_trials <= n_trials:
            return self.trials_per_min
        else: # self.n_trials > n_trials
            n_trials_per_block = [x.n_trials for x in self.task_entries]
            n_trials_cumul = np.cumsum(n_trials_per_block)
            partial_block_idx = np.nonzero(n_trials_cumul > n_trials)[0][0]
            print partial_block_idx, 
            n_trials_needed_from_last_block = n_trials - n_trials_cumul[partial_block_idx-1]
            length = sum([x.length for x in self.task_entries[:partial_block_idx]])
            length += self.task_entries[partial_block_idx].get_length(n_trials_needed_from_last_block)
            return float(n_trials) / length * 60.

    def __repr__(self):
        return 'Combined blocks: %s' % str(self.date)

    @property
    def date(self):
        return self.task_entries[0].date

    @property                                                                  
    def plot_ticklabel(self):                                                                                                                                                                        
        task_types = map(performance.task_type, self.task_entries)
        return '%s\n%s\n%s\n%s\n%d trials\n' % ([x.id for x in self.task_entries], task_types, self.decoder_name, self.date.strftime('%m/%d, %I:%M'), self.n_trials)

starting_block = 2309
def get_all_kf_blocks():
    try:
        blocks_with_kf_decoders = performance.get_kf_blocks_after(starting_block, subject__name__startswith='C')
        blocks_with_kf_decoders = map(performance._get_te, blocks_with_kf_decoders)
        kf_blocks = []
        for block in blocks_with_kf_decoders:
            print block
            # ignore blocks with no HDF files
            if dbfn.get_hdf_file(block.id) is None: 
                continue

            # ignore blocks with fewer than 100 rewards
            if block.n_rewards < 100:
                continue

            if block.n_trials > 100:
                kf_blocks.append(block)
        return kf_blocks
    except:
        import traceback
        traceback.print_exc()
    finally:
        import tables
        tables.file.close_open_files()

def get_blocks_with_ppfdecoder():
    try:
        blocks_with_ppf_decoders = performance.get_ppf_blocks_after(starting_block, subject__name__startswith='C')
        blocks_with_ppf_decoders = map(performance._get_te, blocks_with_ppf_decoders)
        ppf_blocks = []
        for block in blocks_with_ppf_decoders:
            print block
            if dbfn.get_hdf_file(block.id) is None: continue
            if block.n_trials > 100:
                ppf_blocks.append(block)
        return ppf_blocks
    except:
        import traceback
        traceback.print_exc()
    finally:
        import tables
        tables.file.close_open_files()


def get_all_ppf_blocks():
    ppf_decoder_blocks = get_blocks_with_ppfdecoder()
    ppf_blocks = filter(lambda x: not hasattr(x, 'feedback_rate'), ppf_decoder_blocks)
    ppflf_blocks = filter(lambda x: hasattr(x, 'feedback_rate') and hasattr(x, 'task_update_rate') and x.feedback_rate == 10 and x.task_update_rate==60, ppf_decoder_blocks)
    ppflc_blocks = filter(lambda x: hasattr(x, 'feedback_rate') and hasattr(x, 'task_update_rate') and x.feedback_rate == 10 and x.task_update_rate==10, ppf_decoder_blocks)

    ppflc_blocks += [CombinedTaskEntry([2634, 2635]), CombinedTaskEntry([2782, 2783])]
    ppflf_blocks += [CombinedTaskEntry([2620, 2623]), CombinedTaskEntry([2714, 2716]), CombinedTaskEntry([2798, 2806])]
    ppf_blocks += [CombinedTaskEntry([2557, 2559, 2560]), CombinedTaskEntry([2691, 2693])] #, CombinedTaskEntry([2717, 2720])]
    
    # Exclude certain blocks
    ppf_blocks = filter(lambda x: x.id not in [2708], ppf_blocks)

    return ppf_blocks, ppflf_blocks, ppflc_blocks

def task_type(te):
    if te.decoder_type == 'KF':
        return 'KF'
    elif te.decoder_type == 'PPF':
        if not hasattr(te, 'feedback_rate'):
            return 'PPF'
        elif te.task_update_rate == 10:
            return 'LC'
        elif te.task_update_rate == 60:
            return 'LF'

def pair_data(set1, set2):
    from collections import defaultdict, OrderedDict
    import datetime
    data = defaultdict(list)
    type1 = task_type(set1[0])
    type2 = task_type(set2[0])
    for s in set1 + set2:
        date = datetime.date(s.date.year, s.date.month, s.date.day)
        data[date].append(s)

    keys = np.sort(data.keys())
    for date in keys:
        data[date].sort(key=lambda x: x.id)

    pairs = []
    for date in keys:
        stuff = data[date]
        if len(stuff) == 1:
            continue
        block_types = map(task_type, stuff)
        if len(np.unique(block_types)) == 2:
            while len(stuff) > 1 and len(np.unique(block_types)) > 1:
                # get one of each type of block and stuff into pairs
                idx1 = block_types.index(type1)
                idx2 = block_types.index(type2)
                pairs.append( (stuff[idx1], stuff[idx2]) )
                stuff.pop(max(idx1, idx2))
                stuff.pop(min(idx1, idx2))
                block_types = map(task_type, stuff)
            pass
        
    p0 = dbfn.TaskEntrySet([])
    p0.task_entries = [x[0] for x in pairs]
    p1 = dbfn.TaskEntrySet([])
    p1.task_entries = [x[1] for x in pairs]

    paired_data = p0, p1
    #paired_data = dbfn.TaskEntrySet([x[0].id for x in pairs]), dbfn.TaskEntrySet([x[1].id for x in pairs])
    return paired_data

def analyze_pairwise_perf(*block_sets):
    #measures=['reach_time', 'ME']
    measures = dict(reach_time=lambda block_set, n_trials: [np.mean(x.reach_time[:k]) for x, k in izip(block_set.task_entries, n_trials)],
        ME=lambda block_set, n_trials: [np.mean(x.ME[:k]) for x, k in izip(block_set.task_entries, n_trials)],
        perc_correct=lambda block_set, n_trials: [x.get_perc_correct(k) for x, k in izip(block_set.task_entries, n_trials)],
        trials_per_min=lambda block_set, n_trials: [x.get_trials_per_min(k) for x, k in izip(block_set.task_entries, n_trials)],
    )
    #measures=['reach_time', 'perc_correct', 'trials_per_min', 'ME']
    n_trials = np.vstack(map(lambda x: x.n_trials, block_sets)).min(axis=0)
    stuff = dict()
    for measure in measures:
        print measure
        perf = []
        # match the number fo trials
        for block_set in block_sets:
            block_set_perf = measures[measure](block_set, n_trials)
            print block_set.name, block_set_perf
            perf.append(block_set_perf)
        stuff[measure] = perf
        print 'unpaired test', ttest_ind(*perf)
        print 'paired test', ttest_rel(*perf)
        print
    return stuff

# ppf_blocks, ppflf_blocks, ppflc_blocks = get_all_ppf_blocks()
def get_perf(kf_blocks, ppflc_blocks, ppflf_blocks, ppf_blocks):
    kf_ppflc_perf = analyze_pairwise_perf(*pair_data(kf_blocks, ppflc_blocks))
    ppflc_ppflf_perf = analyze_pairwise_perf(*pair_data(ppflc_blocks, ppflf_blocks))
    ppflf_ppf_perf = analyze_pairwise_perf(*pair_data(ppflf_blocks, ppf_blocks))

    return kf_ppflc_perf, ppflc_ppflf_perf, ppflf_ppf_perf

def plot(kf_ppflc_perf, ppflc_ppflf_perf, ppflf_ppf_perf):
    fig = plt.figure(figsize=(12, 8))
    axes = plotutil.subplots(4, 3, hold=True)

    # plot KF, PPFLC
    signs = [-1, +1, +1, -1]
    for k, key in enumerate(['trials_per_min', 'ME', 'reach_time', 'perc_correct']):
        axes[k,0].scatter(1*np.ones(len(kf_ppflc_perf[key][0])), kf_ppflc_perf[key][0])
        axes[k,0].scatter(2*np.ones(len(kf_ppflc_perf[key][0])), kf_ppflc_perf[key][1])
        for a, b in izip(*kf_ppflc_perf[key]):
            axes[k,0].plot([1., 2.], [a, b], color='black')
        axes[k,0].set_xticks([1., 2.])
        axes[k,0].set_xticklabels(['KF', 'PPF-LC'])
        plotutil.set_xlim(axes[k,0], axis='y')
        plotutil.ylabel(axes[k,0], key)
        h, p_val = ttest_rel(kf_ppflc_perf[key][0], kf_ppflc_perf[key][1])
        print h, p_val
        if signs[k]*h > 0:
            p_val = p_val/2
        else:
            pass
        plotutil.xlabel(axes[k,0], 'p=%g, N=%d' % (p_val, len(kf_ppflc_perf[key][0])))

        # plot PPFLC, PPFLF
        axes[k,1].scatter(1*np.ones(len(ppflc_ppflf_perf[key][0])), ppflc_ppflf_perf[key][0])
        axes[k,1].scatter(2*np.ones(len(ppflc_ppflf_perf[key][0])), ppflc_ppflf_perf[key][1])
        for a, b in izip(*ppflc_ppflf_perf[key]):
            axes[k,1].plot([1., 2.], [a, b], color='black')
        axes[k,1].set_xticks([1., 2.])
        axes[k,1].set_xticklabels(['PPF-LC', 'PPF-LF'])
        plotutil.set_xlim(axes[k,1], axis='y')
        h, p_val = ttest_rel(ppflc_ppflf_perf[key][0], ppflc_ppflf_perf[key][1])
        print h, p_val
        if signs[k]*h > 0:
            p_val = p_val/2
        else:
            pass
        plotutil.xlabel(axes[k,1], 'p=%g, N=%d' % (p_val, len(ppflc_ppflf_perf[key][0])))

        # plot PPFLF, PPF
        axes[k,2].scatter(1*np.ones(len(ppflf_ppf_perf[key][0])), ppflf_ppf_perf[key][0])
        axes[k,2].scatter(2*np.ones(len(ppflf_ppf_perf[key][0])), ppflf_ppf_perf[key][1])
        for a, b in izip(*ppflf_ppf_perf[key]):
            axes[k,2].plot([1., 2.], [a, b], color='black')
        axes[k,2].set_xticks([1., 2.])
        axes[k,2].set_xticklabels(['PPF-LF', 'PPF'])
        plotutil.set_xlim(axes[k,2], axis='y')
        h, p_val = ttest_rel(ppflf_ppf_perf[key][0], ppflf_ppf_perf[key][1])
        print h, p_val
        if signs[k]*h > 0:
            p_val = p_val/2
        else:
            pass
        plotutil.xlabel(axes[k,2], 'p=%g, N=%d' % (p_val, len(ppflf_ppf_perf[key][0])))

def get_metrics(block_set):
    '''
    trials = List of measures, blocks, trials
    blocks = list of measures, block averages
    '''

    measures=['reach_time', 'perc_correct', 'trials_per_min', 'ME']
    trials = []
    blocks = []
    for measure in measures:
        #print measure
        bp = getattr(block_set, measure)
        block_set_perf = map(np.mean, getattr(block_set, measure))
        #print block_set.name, block_set_perf
        trials.append(bp)
        blocks.append(block_set_perf)
    return trials, blocks

def analyze_perf(set1,set2):
    measures=['reach_time', 'perc_correct', 'trials_per_min', 'ME']
    trials1,blocks1 = get_metrics(set1)
    trials2,blocks2 = get_metrics(set2)
    pvals_unpaired = [ttest_ind(blocks1[i], blocks2[i]) for i, measure in enumerate(measures)]
    pvals_paired = [ttest_rel(blocks1[i], blocks2[i]) for i, measure in enumerate(measures)]
    return blocks1, blocks2, pvals_unpaired, pvals_paired

def plot_mean_comparisons(sets, pairs, labels):
    measures=['reach time', 'percent correct', 'trials per min', 'movement error']
    for m in range(len(measures)):
        ax=plt.subplot(1,len(measures),m+1)
        xticks = range(1,5)
        pvals = []
        data = [np.mean(get_metrics(set)[1][m]) for set in sets]
        plt.bar(range(1,5),data)
        for i, pair in enumerate(pairs):
            b1, b2, p_u, p_p = analyze_perf(pair[0],pair[1])
            p = p_u
            pvals = pvals +[p]
        dmax = np.max(data)
        dmin = np.min(data)
        lh = dmax+(.2*(dmax-dmin))
        lhall = [(lh+(dmax-dmin)*.1*i)for i in range(len(pairs))]
        for i,ln in enumerate(lhall):
            if i==len(lhall)-1:
                plt.hlines(ln+ln*.05,xticks[0]+.4,xticks[-1]+.4)
                pstr = 'p='+str(np.round(pvals[i][m][1],decimals=3))
                if pvals[i][m][1]<=.05: pstr = '* '+pstr
                plt.text(2.2, ln+ln*.05+(dmax-dmin)*.05, pstr)
            else:
                plt.hlines(ln,xticks[i]+.4,xticks[i+1]+.4)
                pstr = 'p='+str(np.round(pvals[i][m][1],decimals=3))
                if pvals[i][m][1]<=.05: pstr = '* '+pstr
                plt.text(.5+xticks[i], ln+(dmax-dmin)*.05, pstr)
        plt.title(measures[m])
        plt.xlim((0,6))
        plt.ylim((plt.axis()[2], plt.axis()[3]+.1*(plt.axis()[3]-plt.axis()[2])))
        plt.xticks(np.array(xticks)+.4, labels)
    plt.show()

def plot_paired_comparisons(pairs, labels):
    measures=['reach time', 'percent correct', 'trials per min', 'movement error']
    for m in range(len(measures)):
        ax=plt.subplot(1,len(measures),m+1)
        f = ['ko-', 'ro-', 'bo-']
        xticks = [1]
        dmax = 0
        dmin = 1000
        pvals = []
        for i, pair in enumerate(pairs):
            b1, b2, p_u, p_p = analyze_perf(pair[0],pair[1])
            p = p_p
            pvals = pvals +[p]
            data = np.array([b1[m], b2[m]])
            x = [xticks[-1], xticks[-1]+1]
            xticks=xticks+[x[1]]
            plt.plot(x,data,f[i])
            if np.max(data)>dmax: dmax = np.max(data)
            if np.min(data)<dmin: dmin = np.min(data)
        lh = dmax+(.2*(dmax-dmin))
        lhall = [(lh+(dmax-dmin)*.1*i)for i in range(len(pairs))]
        for i,ln in enumerate(lhall):
            plt.hlines(ln,xticks[i],xticks[i+1])
            pstr = 'p='+str(np.round(pvals[i][m][1],decimals=3))
            if pvals[i][m][1]<=.05: pstr = '* '+pstr
            plt.text(.25+xticks[i], ln+(dmax-dmin)*.05, pstr)
        plt.title(measures[m])
        plt.xlim((xticks[0]-1,xticks[-1]+1))
        plt.ylim((plt.axis()[2], plt.axis()[3]+.1*(plt.axis()[3]-plt.axis()[2])))
        plt.xticks(xticks, labels)
    plt.show()

def ppf_summary_plots():
    kf_blocks = get_all_kf_blocks()
    ppf_blocks, ppflf_blocks, ppflc_blocks = get_all_ppf_blocks()
    kf_ppflc_perf, ppflc_ppflf_perf, ppflf_ppf_perf = get_perf(kf_blocks, ppflc_blocks, ppflf_blocks, ppf_blocks)
    plot(kf_ppflc_perf, ppflc_ppflf_perf, ppflf_ppf_perf)
    plt.savefig('/home/helene/ppf_vs_kf.png')
    plt.show()


def _get_te(blocks):
    task_entries = []
    for block in blocks:
        if np.iterable(block):
            task_entries.append(CombinedTaskEntry(block))
        else:
            task_entries.append(performance._get_te(block))

    return task_entries

def proc_pairs(pairs, **kwargs):
    pairs = map(_get_te, pairs)
#    pairs = map(lambda x: tuple([_get_te(y) for y in x]), pairs)

    p0 = dbfn.TaskEntrySet([])
    p0.task_entries = [x[0] for x in pairs]
    p1 = dbfn.TaskEntrySet([])
    p1.task_entries = [x[1] for x in pairs]

    perf = analyze_pairwise_perf(p0, p1)
    plot_pair(perf, **kwargs)

def plot_pair(perf, labels=['KF', 'PPF']):
    fig = plt.figure(figsize=(4, 8))
    axes = plotutil.subplots(4, 1, hold=True)

    # plot KF, PPFLC
    signs = [-1, +1, +1, -1]
    for k, key in enumerate(['trials_per_min', 'ME', 'reach_time', 'perc_correct']):
        axes[k,0].scatter(1*np.ones(len(perf[key][0])), perf[key][0])
        axes[k,0].scatter(2*np.ones(len(perf[key][0])), perf[key][1])
        for a, b in izip(*perf[key]):
            axes[k,0].plot([1., 2.], [a, b], color='black')
        axes[k,0].set_xticks([1., 2.])
        axes[k,0].set_xticklabels(labels)
        plotutil.set_xlim(axes[k,0], axis='y')
        plotutil.ylabel(axes[k,0], key)
        h, p_val = ttest_rel(perf[key][0], perf[key][1])
        print h, p_val
        if signs[k]*h > 0:
            p_val = p_val/2
        else:
            pass
        plotutil.xlabel(axes[k,0], 'p=%g, N=%d' % (p_val, len(perf[key][0])))
