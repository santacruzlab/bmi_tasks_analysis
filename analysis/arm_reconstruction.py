'''
Methodds to reconstruct arm posture from noisy motion tracker data
'''

import tables
import math
import numpy as np
import random
import time

def findFirsts(data):
    ''' Find the earliest index of good data for each marker.
        data = time x nmarkers x 4 array
        earliest = nmarkers x 1 array
        '''
    earliest = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        earliest[i] = np.nonzero(np.logical_and(data[:,i,-1]!=4, data[:,i,-1]>0))[0][0]
    return earliest

def findMods(data, firsts):
    '''
    Return the modulus value (0-3) for non-interpolated data for each marker. Ex: mod=2
    means that index 2,6,10,etc are non-interpolated for that marker.
    data = time x nmarkers x 4 array
    firsts = nmarkers x 1 array of first instance of non-interpolated data for each marker
    mods = nmarkers x 1 array
    '''
    mods = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        mods[i]=np.mod(firsts[i],4)
    return mods

def genInds(l,m):
    return np.array(range(int(m),l,4))

def ds(data):
    '''
    Downsamples raw motiontracker data from 240hz to 60 hz. Gets rid of all interpolated
    data points and replaces missing data with nans. Also cuts off beginning of data until
    first good value appears for any marker.

    data = time x nmarkers x 4 array
    newdata = ~time/4 x nmarkers x 3 array
    '''
    f=findFirsts(data)
    data=data[np.min(f):]
    f=findFirsts(data)
    m=findMods(data,f)
    l=len(genInds(len(data),m[0]))-1
    newdata = np.zeros([l,data.shape[1],3])
    for i in range(data.shape[1]):
        print i
        t=data[genInds(len(data),m[i]),i,:]
        mask=t[:,-1]==-1
        t[mask,:3]=np.nan
        newdata[:,i,:]=t[:l,:3]

    return newdata

def plotFrame(i,w,e,s,data,ax,fig):
    '''
    3D plot of reconstructed arm position and marker positions for a given frame.

    i= frame index
    w= wrist xyz positions (time x 3 array)
    e = elbow xyz positions
    s = shoulder xyz positions
    data = downsampled raw marker data (must be same length as w,e, and s)
    ax = figure axis
    fig = figure
    '''
    x=np.array([w[i,0],e[i,0],s[i,0]])
    y=np.array([w[i,1],e[i,1],s[i,1]])
    z=np.array([w[i,2],e[i,2],s[i,2]])

    shx=x
    shy=y
    shz=np.array([200,200,200])

    clrs=['r','g','y','b']
    ax.cla()
    ax.plot(x,y,z,linewidth=5,c='k')
    ax.plot(shx,shy,shz,linewidth=5,c='grey')
    for j in range(data.shape[1]):
        ax.scatter(data[i,j,0],data[i,j,1],data[i,j,2],c=clrs[np.mod(j,len(clrs))],s=15)
    ax.set_xlim(-50,250)
    ax.set_ylim(-150,150)
    ax.set_zlim(-100,200)
    fig.canvas.draw()

def genMovieInds(start,seconds):
    '''
    generates indices of data frames to use for genMovieInds
    start = starting frame index
    seconds = movie length in seconds
    '''
    numFrames=int(seconds*30)
    stop=start+(numFrames-1)*2+1
    return np.array(range(start,stop,2))

def saveMovieFrames(data, start, seconds,w,e,s,ax,fig):
    '''
    Generates plots for movie frames and saves them as png files with sequential nframes
    data= downsampled marker data
    start = starting data frame
    seconds = length of movie
    w = wrist positions
    e = elbow positions
    s = shoulder positions
    ax = axis handle
    fig = figure handle
    '''
    path='/Users/helene/Dropbox/Docs/'
    inds=genMovieInds(start,seconds)
    nframes=len(inds)
    for j, i in enumerate(inds):
        name='frame' + str(j).zfill(3)+'.png'
        plotFrame(j,w[inds],e[inds],s[inds],data[inds],ax,fig)
        fig.savefig(path+name,dpi=150)


def findWrist(fa_data, elbow):
    '''
    Estimates wrist positions using elbow positions and positions of forearm markers.
    Does it by calculating the endpoint of a line of fixed length (135 mm) starting at
    elbow and pointing in the direction of the average of all available wrist-forearm marker
    vectors.
    fa_data = downsampled marker data for forearm markers (time x nmarkers x 3 array)
    elbow = time x 3 array of elbow positions (must be same length as fa_data)
    returns time x 3 array of wrist positions (unknown positions are nans)
    '''
    t=fa_data-elbow[:,None,:]
    direction=np.zeros([len(fa_data),3])
    for i in range(len(fa_data)):
        d=np.sum(t[i][~np.isnan(t[i,:,0]),:],axis=0)
        if np.all(d==0):
            direction[i]=np.array([np.nan,np.nan,np.nan])
        else:
            direction[i]=d
    length=np.ones([len(fa_data)])*135
    mag = np.sqrt(direction[:,0]**2+direction[:,1]**2+direction[:,2]**2)
    return elbow+direction/mag[:,None]*length[:,None]


def chooseRandom(downsample, num):
    '''
Function: chooseRandom, randomly chooses 1000 frames to use
Input/Output: None
Parameters: downsample, a downsampled array holding real data
Return: frames, a 1000x16x3 array of random frames

'''
    nMarkers = downsample.shape[1]
    cleanFrames = []
    #if the sum of the nans mask for a given frame is zero then add it to a list of nans free frames
    for x in range(downsample.shape[0]):
        if(np.isnan(downsample[x,:,0]).sum() == 0):
            cleanFrames.append(x)
    np.random.shuffle(cleanFrames)
    #Use the first num of theses shuffled indices to choose num frames
    frames= downsample[cleanFrames[:num],:, 0:3]
    return frames



def findElbow(frames, allFrames):
    '''
Function: findElbow, generates an estimated location of the joint
Input/Output: None
Parameters: frames, a randomly chosen nframes x nMarkers x 3 array holding the positions of the marker per frame
            allFrames, all of the data available from the trial
Returns: distances, a nMarker vector of the mean of the each frame's distance from the marker to elbow estimate
        pos, the estimated location of the elbow position for each frame of data.

'''
    #initialize necessary values
    nframes = len(frames)
    nMarkers = frames.shape[1]

    '''
    Function: jointCost, determines the joint cost (mean variance)
    Input/Output: None
    Parameters: frames, a 1000x16x3 array of randomly chosen frames
                jointPos, a 1000x3 array of joint positions
    Returns: meanDist, a 16 element vector containing the mean distance to the joint from each marker
            sigma, a 16 element vector containing the variance within the distances 
    
    '''
    
    def jointCost(frames, jointPos):  
        rawDist = np.zeros([nframes, nMarkers])    
        for x in range(nMarkers):
            #Use the distance formula to find the distance between the marker and joint
            #if it is nans, then the distance will be nans too. 
            rawDist[:,x] = np.sqrt((frames[:, x, 0]-jointPos[:,0])**2+(frames[:, x, 1]-jointPos[:,1])**2+(frames[:, x, 2]-jointPos[:,2])**2)     
        #define a 16 element vector to store mean distances and the variance
        meanDist = np.zeros([nMarkers])
        sigma = np.zeros([nMarkers])
        #for each marker, find the mean distance between the marker and joint, along with the variance (std**2)
        for i in range(nMarkers):
            meanDist[i] = np.mean(rawDist[np.nonzero(~np.isnan(frames[:, i, 0])),i])
            sigma[i]= np.std(rawDist[np.nonzero(~np.isnan(frames[:, i, 0])),i])**2
        return meanDist, sigma
        
    '''
    Function: addDistPen, adds a penalty for distance, encouraging to algorithm to not find the
                arbitrary answer when minimizing the variance, which is to just go far away. 
    Input/Output: none
    Parameters: frames, a 1000x16x3 array of random frames
                jointPos, a 1000x3 array of joint positions
    Return: result, 
    '''
    def addDistPen(jointPos):
        dist, sigma = jointCost(frames, jointPos.reshape(nframes, 3))
        #distance penalty, will need to be adjusted and experimented with
        alpha = .1
        result = (dist*alpha + sigma).sum()        
        return result

    #Used pretty much straight from James' motion.py, added tolerance and got rid of the option to display the outcome   
    from scipy.optimize import minimize
    init = np.zeros((nframes, 3)).ravel()
    result = minimize(addDistPen, init, tol=.001)
    dist, sigma = jointCost(frames, result.x.reshape(nframes,3))
    pos = result.x.reshape(nframes, 3)
    
    #with the marker data and the joint estimates, return the mean distances for each of the 16 markers
    distances = np.zeros([nMarkers])
    distances = jointCost(frames,pos)[0]
    
    # initialize array to store elbow locations for every data frame and set the values to nans so its obvious if they aren't changed
    elbow_locations=np.zeros([allFrames.shape[0],3]) 
    elbow_locations[:,:] = np.nan

    # for each frame, get the marker position data
    for i in range(allFrames.shape[0]):
        if np.mod(i,1000)==0:
            print i
        frame=allFrames[i] 
        '''
        Function: distError, calculate the distance from the input point to each of the marker position, subtract the distance,
                  square it (to account for negatives), and sum to get an error value 
        Input/Output: None
        Parameters: elbow, a 3 element array [x,y,z]
        Returns: an error value correspoinding to the distance
        '''
        #define distError inside of the for loop to use i as an input without messing up the inputs for the format of the function
        def distError(elbow):
            #caclulate the distance from this point to each of the marker positions, subtract the known distance, square and sum
            d=(np.sqrt((elbow[0]-frame[:,0])**2 + (elbow[1]- frame[:,1])**2 + (elbow[2]-frame[:,2])**2)-dist)**2
            error = np.sum(d[~np.isnan(d)])
            if error==0:
                return np.nan
            else:
                return error
        # Fit the point that best satisfies the distance constraints.
        elbow_locations[i] = minimize(distError, np.zeros([3]), tol=.001).x
        # if the minimize function returned 0s it means that the error function returned nan, so change it to nan as well
        if np.any(elbow_locations[i]==0):
            elbow_locations[i]=np.nan
    return elbow_locations

def findShoulder(frames, allFrames):
    '''
    Similar to findElbow above but estimates a single fixed shoulder position across all frames
    '''

    #initialize necessary values
    nframes = len(frames)
    nMarkers = frames.shape[1]

    '''
    Function: jointCost, determines the joint cost (mean variance)
    Input/Output: None
    Parameters: frames, a 1000x16x3 array of randomly chosen frames
                jointPos, a 1000x3 array of joint positions
    Returns: meanDist, a 16 element vector containing the mean distance to the joint from each marker
            sigma, a 16 element vector containing the variance within the distances 
    
    '''
    
    def jointCost(frames, jointPos):  
        rawDist = np.zeros([nframes, nMarkers])    
        for x in range(nMarkers):
            #Use the distance formula to find the distance between the marker and a single non-moving joint position
            #if it is nans, then the distance will be nans too. 
            rawDist[:,x] = np.sqrt((frames[:, x, 0]-jointPos[0])**2+(frames[:, x, 1]-jointPos[1])**2+(frames[:, x, 2]-jointPos[2])**2)     
        #define a 16 element vector to store mean distances and the variance
        meanDist = np.zeros([nMarkers])
        sigma = np.zeros([nMarkers])
        #for each marker, find the mean distance between the marker and joint, along with the variance (std**2)
        for i in range(nMarkers):
            meanDist[i] = np.mean(rawDist[np.nonzero(~np.isnan(frames[:, i, 0])),i])
            sigma[i]= np.std(rawDist[np.nonzero(~np.isnan(frames[:, i, 0])),i])**2
        return meanDist, sigma
        
    '''
    Function: addDistPen, adds a penalty for distance, encouraging to algorithm to not find the
                arbitrary answer when minimizing the variance, which is to just go far away. 
    Input/Output: none
    Parameters: frames, a 1000x16x3 array of random frames
                jointPos, a 1000x3 array of joint positions
    Return: result, 
    '''
    def addDistPen(jointPos):
        dist, sigma = jointCost(frames, jointPos)
        #distance penalty, will need to be adjusted and experimented with
        alpha = .1
        result = (dist*alpha + sigma).sum()        
        return result

    #Used pretty much straight from James' motion.py, added tolerance and got rid of the option to display the outcome   
    from scipy.optimize import minimize
    init = np.zeros([3])
    result = minimize(addDistPen, init, tol=.001)
    dist, sigma = jointCost(frames, result.x)
    return result.x


def accuracy(genElbowPos, realData):
    '''
Function: accuracy- a funciton to determine the accuracy of the findElbow function
Input/Output: none
Parameters: genElbowPos: The elbow positions generated by the findElbow function
            realData: The real elbow positions genenrated in the fakeArmData function
Returns: a list of distances between the generated position and real position in order 
        judege accruacy. For 100 percent accuracy, the distances should be 0.
'''
    numFrames = genElbowPos.shape[0]
    dist = np.zeros(numFrames)
    dist = np.sqrt((genElbowPos[:, 0]-realData[:,0])**2+(genElbowPos[:, 1]-realData[:,1])**2+(genElbowPos[:, 2]-realData[:,2])**2)
    #average the values to get a cleaner output, the lower the number, the higher the accuracy
    n = len(dist)
    stdError = np.std(dist)/np.sqrt(n)
    return np.mean(dist[~np.isnan(dist)]) , stdError


'''
For the fakeArmData function I added a noise input which is the standard deviation
in mm of a noise term that gets added to the marker positions that the function
outputs. This allows us to test fake data with varying levels of noise added in to
see how accurate our processing is under more realistic conditions.

I measured the monkey's arm and got approximately 15 cm for the upper arm length
and 20 cm for the forearm, hence the default values for r1 and r2.
'''

'''
Function: fakeArmData, generates realistic fake arm data to judge the accuracy of the findElbow function. 
Inputs/Outputs: none
Parameters: numFrames, the number of frames desired.
            r1, the length of the upper arm (from shoulder to elbow) (in cm)
            r2, the length of the lower arm (from elbow to hand) (in cm)
            noise, a value that can add noise to the data
Returns: markerData+ noiseValues, the positions of the generated markers to feed to findElbow
        armPos[:,0,:], the true elbow positions that serve as a bench mark for the accuracy of findElbow
'''
def fakeArmData(numFrames, noise, percentNans, r1=150,r2=200):
    #create the array to store the elbow and hand positions in,
    #the shape is numFramesx2x3 --> 0 is for elbow pos, and 1 is for the hand pos
    armPos = np.zeros([numFrames,2,3])
    markerData = np.zeros([numFrames, 6, 3])

    #Set shoulder to be fixed at (0,0,0)
    #Get random angles between 0 and 2pi, and u between -1 and 1
    angl = np.random.uniform(0,2*np.pi,numFrames)
    u = np.random.uniform(-1,1,numFrames)

    #find the elbow position
    armPos[:,0,0]= np.cos(angl)*np.sqrt(1-u**2)*r1 
    armPos[:,0,1]= np.sin(angl)*np.sqrt(1-u**2)*r1
    armPos[:,0,2]= u*r1
    
    #for the hand position, calculate the random point on the sphere the same way as if the center was the origin, 
    #but then after, add the x y and z coordinates to the point to get the correct frame of reference
    #create lists of random numbers as long as numFrames, for angl between 0 and 2pi, and for u between -1 and 1
    angl = np.random.uniform(0,2*np.pi,numFrames)
    u = np.random.uniform(-1,1,numFrames)

    armPos[:, 1, 0]= np.cos(angl)*np.sqrt(1-u**2)*r2+ armPos[:,0,0]  
    armPos[:, 1, 1]= np.sin(angl)*np.sqrt(1-u**2)*r2+ armPos[:,0,1]
    armPos[:, 1, 2]= u*r2 +armPos[:,0, 2]

    #now that we have the shoulder, elbow and hand positions for each time frame, fill in the marker data
    #Need 3 evenly spaced points on a line, for each marker(6)
    
    #find coordinates evenly spaced on the line given (0,0,0) and the elbow position
    markerData[:,0,:] = .25*(armPos[:,0,:])
    markerData[:,1,:] = .5*(armPos[:,0,:])
    markerData[:,2,:] = .75*(armPos[:,0,:])
    
    #find coordinates evenly spanced in the line given elbow positions and hand positions
    markerData[:,3,:] = .25*(armPos[:, 1, :] - armPos[:,0,:]) +armPos[:,0,:]       
    markerData[:,4,:] = .5*(armPos[:, 1, :] - armPos[:,0,:]) +armPos[:,0,:]
    markerData[:,5,:] = .75*(armPos[:, 1, :] - armPos[:,0,:]) +armPos[:,0,:]

    # Create noise array and add it to position values if noise input is specified
    if noise is None:
        noise_terms = np.zeros([markerData.shape[0],markerData.shape[1],markerData.shape[2]])
    else:
        noise_terms = np.random.randn(markerData.shape[0],markerData.shape[1],markerData.shape[2])*2*noise-noise

    def addNans(percentNans):
        if(percentNans == 0):
            pass
        else:
            #calculate the number of nans to add based on the numFrames and percentage inputed by user
            numData = numFrames*6
            numNans = numData*percentNans
            numNans = int(numNans)
            #create a list of random frames as long as numNans
            nanFrames = np.random.randint(0, numFrames,numNans)
            nanMarkers = np.random.randint(0, 6, numNans)
            for i in range(numNans):
                markerData[nanFrames[i], nanMarkers[i], :]= np.nan
    addNans(percentNans)
    return markerData+noise_terms, armPos[:,0,:]

def simulatedData(nTotalFrames, nSubset, percentNans=0, noise=None):
    allFakeMarkerData, realElbowPos = fakeArmData(nTotalFrames,noise, percentNans)
    fakeSubset = chooseRandom(allFakeMarkerData, nSubset)
    generatedElbowPos = findElbow(fakeSubset, allFakeMarkerData)
    return accuracy(generatedElbowPos, realElbowPos)

def simulatedData2(nTotalFrames, nSubset, percentNans=0, noise=None):
    allFakeMarkerData, realElbowPos = fakeArmData(nTotalFrames,noise, percentNans)
    fakeSubset = chooseRandom(allFakeMarkerData, nSubset)
    errs=np.zeros(6)
    stderrs=np.zeros(6)
    generatedElbowPos = findElbow(fakeSubset, allFakeMarkerData)
    indsall=[]
    for i in range(6):
        d,inds=sortData(allFakeMarkerData,i)
        errs[i], stderrs[i]=accuracy(generatedElbowPos[inds],realElbowPos[inds])
        indsall = indsall +[inds]
    return errs, stderrs

def sortData(data,numnan):
    rownums=np.sum(np.isnan(data[:,:,0]),axis=1)
    inds = np.nonzero(rownums==numnan)[0]
    print numnan, len(inds)
    return data[inds], inds

def noiseData():
    noiseErrorData = np.zeros(10)
    stdError = np.zeros(10)
    for i in range(10):
        noiseErrorData[i], stdError[i] = simulatedData(10000, 100, 0, .5*i)
        if(i % 10 == 0):
            print i
    return noiseErrorData, stdError

def nansData():
    stdError = np.zeros(10)
    nansData = np.zeros(10)
    for i in range(10):
        nansData[i], stdError[i] = simulatedData(10000,100, .05*i, 0)
        print i
    return nansData, stdError
    
def realData(downsample):
    subset = chooseRandom(downsample, 500)
    genElbowPos = findElbow(subset, downsample)
    return genElbowPos

