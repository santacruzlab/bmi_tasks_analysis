import dbfunctions

kf_ppf_pairs = [(2310, 2315), (2325, 2330), (2380, 2378), (2424, 2425), (2446, 2440), (2441, 2445), (2457, 2455), (2465, 2469), (2613, 2609), (2650, 2644), (2661, 2662), (2674, 2670), (2689, (2691,2693)), (2749, 2738), ((2791,2795), 2796), (2841, (2837,2838)), (2857, 2858), (2860, 2859), (2869, (2871,2872,2873))]

kf_lc_pairs = [(2325, 2332), (2424, 2426), (2446, 2442), (2457, 2454), (2465, (2468,2470)), (2613, 2608), (2650, (2646,2647)), (2661, 2657), (2674, 2668), (2689, 2690), (2749, 2740), ((2791,2795), (2798,2806))]

lc_lf_pairs = [(2468, 2467), (2470, 2471), (2495, 2493), (2556, 2555), (2608, 2610), (2619, (2620,2623)), ((2634,2635), 2631), ((2646,2647), 2645), (2657, 2658), (2668, 2669), (2713, (2714,2716)), (2740, 2739), (2759, 2760), ((2782,2783), 2781)]

lf_ppf_pairs = [(2467, 2469), (2493, 2494), (2573, 2574), (2579, 2578), (2610, 2609), ((2620,2623), 2618), (2631, 2632), (2645, 2644), (2663, 2656), (2658, 2662), (2669, 2670), (2739, 2738), (2781, 2784), (2760, 2761), ((2798,2806), 2796), (2820, 2819), (2824, 2826)]


all_kf = []
for pair in kf_ppf_pairs:
    if isinstance(pair[0], int):
        all_kf = all_kf + [pair[0]]
    else:
        all_kf = all_kf + list(pair[0])

for pair in kf_lc_pairs:
    if isinstance(pair[0], int):
        all_kf = all_kf + [pair[0]]
    else:
        all_kf = all_kf + list(pair[0])

all_kf = list(set(all_kf))
all_kf.sort()


all_ppf = []
for pair in kf_ppf_pairs:
    if isinstance(pair[1], int):
        all_ppf = all_ppf + [pair[1]]
    else:
        all_ppf = all_ppf + list(pair[1])

for pair in lf_ppf_pairs:
    if isinstance(pair[1], int):
        all_ppf = all_ppf + [pair[1]]
    else:
        all_ppf = all_ppf + list(pair[1])

all_ppf = list(set(all_ppf))
all_ppf.sort()

all_lc = []
for pair in kf_lc_pairs:
    if isinstance(pair[1], int):
        all_lc = all_lc + [pair[1]]
    else:
        all_lc = all_lc + list(pair[1])
for pair in lc_lf_pairs:
    if isinstance(pair[0], int):
        all_lc = all_lc + [pair[0]]
    else:
        all_lc = all_lc + list(pair[0])
all_lc = list(set(all_lc))
all_lc.sort()

all_lf = []
for pair in lc_lf_pairs:
    if isinstance(pair[1], int):
        all_lf = all_lf + [pair[1]]
    else:
        all_lf = all_lf + list(pair[1])
for pair in lf_ppf_pairs:
    if isinstance(pair[0], int):
        all_lf = all_lf + [pair[0]]
    else:
        all_lf = all_lf + list(pair[0])
all_lf = list(set(all_lf))
all_lf.sort()



all_kf = [2310, 2325, 2380, 2424, 2441, 2446, 2457, 2465, 2613, 2650, 2661, 2674, 2689, 2749, 2791, 2795, 2841, 2857, 2860, 2869]
all_ppf = [2315, 2330, 2378, 2425, 2440, 2445, 2455, 2469, 2494, 2574, 2578, 2609, 2618, 2632, 2644, 2656, 2662, 2670, 2691, 2693, 2738, 2761, 2784, 2796, 2819, 2826, 2837, 2838, 2858, 2859, 2871, 2872, 2873]
all_lc = [2332, 2426, 2442, 2454, 2468, 2470, 2495, 2556, 2608, 2619, 2634, 2635, 2646, 2647, 2657, 2668, 2690, 2713, 2740, 2759, 2782, 2783, 2798, 2806]
all_lf = [2467, 2471, 2493, 2555, 2573, 2579, 2610, 2620, 2623, 2631, 2645, 2658, 2663, 2669, 2714, 2716, 2739, 2760, 2781, 2798, 2806, 2820, 2824]

def getfns(lst):
    for el in lst:
        print el
    for el in lst:
        print dbfunctions.get_hdf_file(el)[21:]
