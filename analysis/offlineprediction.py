import numpy as np


def create_lagged_arrays(input_data, output_data, nlags):
  ''' Returns a lagged input array for linear regression and the corresponding
  output array with nans trimmed. Left half of input array
  contains features where the input follows the output in time, right side
  the input precedes the output in time. lags argument determines the number
  of positive and negative lags.
  '''

  inputs = np.zeros([input_data.shape[0]+(2*nlags), ((2*nlags)+1)*input_data.shape[1]])
  inputs[:,:] = np.nan

  for i in range((2*nlags)+1):
    inputs[i:i+input_data.shape[0],i*input_data.shape[1]:(i*input_data.shape[1])+input_data.shape[1]] = input_data

  inputs = inputs[np.sum(np.isnan(inputs),axis=1)==0, :]
  outputs = output_data[nlags:-nlags]

  return inputs, outputs

def split_data(data, ngroups):
  ''' Split up data array into n equal size groups for cross-validation. Returned as list.'''
  lengroup = int(np.floor(len(data)/ngroups))
  result=[]
  for i in range(ngroups):
    result = result+[data[i*lengroup:(i*lengroup)+lengroup]]
  return result









def create_bins(hdf, plx, binsize, ts_func):
  # binsize = bin length in second (best if factor of 240 hz)
  # Uses the hdf file to assign bins, returns hdf timestamps and corresponding
  # plexon timestamps

  hdfstart = 0
  hdfend = len(hdf.root.task) - 1
  plxstart = ts_func([hdfstart],'plx')[0]
  plxend = ts_func([hdfend], 'plx')[0]
  nbins = np.floor((plxend-plxstart)/binsize)
  plxbins = np.linspace(plxstart, plxstart+binsize*nbins, nbins+1)
  hdfbins = np.array(ts_func(plxbins, 'hdf'))

  return plxbins, hdfbins


def bin_spikes(plx, bins):

  # bins = bin boundaries

  units = plx.units
  counts = np.zeros([len(bins)-1, len(units)])

  for i in range(len(bins)-1):
    all_spikes = plx.spikes[bins[i]:bins[i+1]].data
    for j, unit in enumerate(units):
      counts[i,j]=len(all_spikes[np.logical_and(all_spikes['chan'] == unit[0],
                      all_spikes['unit'] == unit[1])])
  return counts


def bin_kindata(markerdata, bins):

  # markerdata = n x 3 array for single marker position
  # bins = bin boundaries
  # returns x,y,z position and velocity

  vals = np.zeros([len(bins) - 1, 6])

  # downsample position data and get rid of nans
  for i in range(len(bins)-1):
    bindata = markerdata[bins[i]:bins[i+1]]
    for j in range(3):
      dat = bindata[:,j]
      # fill in mean, nan if all nans
      if np.sum(~np.isnan(bindata))==0:
        vals[i,j] = np.nan
      else:
        vals[i,j] = np.mean(dat[~np.isnan(dat)])
      # get rid of all but leading nans by filling them in with prev. vals
      if i>0 and np.isnan(vals[i,j]):
        vals[i,j] = vals[i-1,j]

  # fill in velocities from positions
  vals[1:,3:] = np.diff(vals[:,:3],axis=0)

  return vals


def fit_linear_model(input, output, numdelays):
  if np.ndim(input)<2:
    input = input[:,None]
  numrows = input.shape[0] - numdelays + 1
  numunits = input.shape[1]

  Y = output[numdelays-1:]

  R = np.zeros([numrows, numunits*numdelays])

  for i in range(numrows):
    for j in range(numunits):
      R[i,j*numdelays:(j+1)*numdelays] = input[i:i+numdelays,j]

  X = np.concatenate((np.ones([numrows,1]), R), axis=1)

  t1 = np.dot(X.T,X)
  t2 = np.linalg.pinv(t1)
  t3 = np.dot(X.T,Y)
  A = np.dot(t2,t3)

  return A

def create_X(input, numdelays):
  if np.ndim(input)<2:
    input = input[:,None]
  numrows = input.shape[0] - numdelays + 1
  numunits = input.shape[1]

  R = np.zeros([numrows, numunits*numdelays])

  for i in range(numrows):
    for j in range(numunits):
      R[i,j*numdelays:(j+1)*numdelays] = input[i:i+numdelays,j]

  return np.concatenate((np.ones([numrows,1]), R), axis=1)



