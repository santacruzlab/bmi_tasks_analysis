"""
Modified for Ripple neural recording system. To load the dataset, we 
use io.BlackrockIO.

@author: yc25258
"""
import numpy as np
import scipy as sp
from neo import io

def create_syncHDF(filename, Ripple_folder):
    #Ripple_folder = 'F:/yc25258/Ripple_example_data/Example12/'
    #filename = 'datafile0012'
    data_filename = Ripple_folder + filename + ".nev"
    
    # Load Ripple files.
    r = io.BlackrockIO(data_filename)
    bl = r.read_block()
    print("File read.")
    
    # Analogsignals for digital events
    # Naming convention
    # 0 - 3  : SMA 1 - 4
    # 4 - 27 : Pin 1 - 24
    # 28 - 29: Audio 1 - 2 
    # Here we use Pin 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19
    signals = bl.segments[0].analogsignals[-1]
    pins_util = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19]) + 3
    rstart = np.array(signals[:, 22 + 3])
    strobe = np.array(signals[:, 20 + 3])
    msgtype = np.array(signals[:, pins_util[8:]])
    rwoNum = np.array(signals[:, pins_util[:8]])
    # Convert to binary numbers (0 ~ 5000 from the recordings)
    msgtype[msgtype<=2500] = 0
    msgtype[msgtype>2500] = 1
    rwoNum[rwoNum<=2500] = 0
    rwoNum[rwoNum>2500] = 1
    rstart[rstart<=2500] = 0
    rstart[rstart>2500] = 1  
    strobe[strobe<=2500] = 0
    strobe[strobe>2500] = 1
    rstart = np.ravel(rstart)
    strobe = np.ravel(strobe)
    # Convert the binary digits into arrays
    MSGTYPE = np.zeros(strobe.shape)
    ROWNUMB = np.zeros(strobe.shape)
    for tp in range(MSGTYPE.shape[0]):
        temp_str = ''
        for i in range(8):
            temp_str += str(int(msgtype[tp, 7-i]))
        MSGTYPE[tp] = int(temp_str, 2)
        temp_str = ''
        for i in range(8):
            temp_str += str(int(rwoNum[tp, 7-i]))
        ROWNUMB[tp] = int(temp_str, 2)
    print("Row Number extracted.")
    
    # Create dictionary to store synchronization data
    hdf_times = dict()
    hdf_times['row_number'] = []          # PyTable row number
    hdf_times['ripple_samplenumber'] = []    # Corresponding Ripple sample number
    hdf_times['ripple_dio_samplerate'] = []  # Sampling frequency of DIO signal recorded by Ripple system
    hdf_times['ripple_recording_start'] = [] # Ripple sample number when behavior recording begins
        
    find_recording_start = np.ravel(np.nonzero(rstart))[0]
    find_data_rows = np.logical_and(np.ravel(np.equal(MSGTYPE,13)),np.ravel(np.greater(strobe,0))) 	
    find_data_rows_ind = np.ravel(np.nonzero(find_data_rows))
    
    rows = ROWNUMB[find_data_rows_ind]	  # row numbers (mod 256)
    
    prev_row = rows[0] 	# placeholder variable for previous row number
    counter = 0 		# counter for number of cycles (i.e. number of times we wrap around from 255 to 0) in hdf row numbers
    
    for ind in range(1,len(rows)):
    	row = rows[ind]
    	cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
    	counter += cycle
    	rows[ind] = counter*256 + row
    	prev_row = row    
    
    # Load data into dictionary
    hdf_times['row_number'] = rows
    hdf_times['ripple_samplenumber'] = find_data_rows_ind
    hdf_times['ripple_recording_start'] = find_recording_start
    hdf_times['ripple_dio_samplerate'] = signals.sampling_rate
    
    # Save syncing data as .mat file
    mat_filename = filename + '_syncHDF.mat'
    #sp.io.savemat(Ripple_folder+mat_filename,hdf_times)
    print(Ripple_folder+mat_filename)
            
    return hdf_times
    