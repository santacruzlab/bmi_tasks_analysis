import re
import numpy as np
import pandas as pd

def parse(ascfile, columns=("timestamp", "x", "y", "area", "status")):
    size = None
    data = []
    messages = []
    last = 0
    for line in ascfile:
        if line.startswith("MSG"):
            messages.append(parse_msg(line))
        if line[0] in "0123456789":
            try:
                d = np.fromstring(line, sep=" ")
                if len(d) == len(columns):
                    if d[0] <= last:
                        print d[0]
                    data.append(d)
                    last = d[0]
            except:
                pass

    names = {}
    codes = np.empty((len(messages), 2))
    for i, (time, msg) in enumerate(messages):
        if msg not in names:
            names[msg] = len(names)
        codes[i] = time, names[msg]

    codes = np.vstack([codes[:-1,0], codes[1:,0], codes[:-1,1]]).T
    data = pd.DataFrame(data, columns=columns).set_index("timestamp")
    data.index = pd.Int64Index(data.index)
    return data, codes, names

def parse_msg(line):
    ts, data = re.match('MSG\s+(\d+)\s+(.*)', line).groups()
    return int(ts), data

def find_reward(names, codes, sequence=("pretrial", "trial", "reward")):
    pattern = np.array([names[s] for s in sequence])
    match = (pattern**2).sum()
    return np.correlate(codes[:,-1], pattern, mode="same") == match

def get_rewarded_chunks(data, times, rewards, perimeter=3):
    chunks = []
    ctime = times[rewards][:,0]
    edges = np.diff(np.convolve(rewards, [1]*perimeter, mode='same'))

    for s, m, e in zip(times[edges==1,0], ctime, times[edges==-1,1]):
        chunk = data.ix[s:e]
        chunk.index = pd.Int64Index(np.array(chunk.index) - m)
        chunks.append(chunk)
    return chunks

def get_peristate_chunks(data, times, mask, offset=500):
    chunks = []
    for t in times[mask][:,0]:
        chunk = data.ix[t-offset:t+offset]
        chunk.index = pd.Int64Index(np.array(chunk.index) - t)
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    import sys
    data, codes, names = parse(open(sys.argv[1]))
    rewards = find_reward(names, codes)

    chunks = get_rewarded_chunks(data, codes[:,:2], rewards)
    #chunks = get_peristate_chunks(data, codes[:,:2], rewards)
    #pts = [d[:,1:3] for d in chunks]