import numpy as np
import matplotlib.pyplot as plt
import sys, glob
import soundfile as sf # <-- read audio
from scipy.spatial import ConvexHull
import librosa # <-- resample function
from scipy import signal # <-- fast convolution function
from IPython.display import Audio # <-- Audio listening in notebook
import copy
import random

class hrtf():
    
    # initialization
    def __init__(self, hrtf_home):
       
        self.hrir_array = []
        self.hrir_sr = 0
        self.hrtf_dir_MIT = hrtf_home
        
        self.ls_az = np.concatenate((np.arange(0, 360, 30),np.arange(15, 360, 30), np.arange(0, 360, 30), np.zeros(2)))
        self.ls_ele = np.concatenate((np.ones(12)* -30, np.zeros(12), np.ones(12)* 30, np.array([90, -90])))
        self.hrir_vector = np.vstack([self.ls_az,self.ls_ele]).transpose()
        pass

    # load hrir array and sample rate
    def load(self):
        self._MIT = glob.glob(self.hrtf_dir_MIT)
        self._MIT.sort()
        
        for i in range(38):
            [HRIR,fs_H] = sf.read(self._MIT[i])
            self.hrir_sr = fs_H
            self.hrir_array.append(HRIR)
    
    # print file names
    def print_files(self):
        print('List of HRTF files:')
        for s in range(len(self._MIT)):
            print('\033[1m' + str(s) +'. ' + '\033[0m' + self._MIT[s][13:]) 
        pass
    
    # diaplay audio files for checking
    def play_files(self):
        for i in range(38):
            print('HRTF' + str(i) + ':')
            display(Audio(self.hrir_array[i].transpose(),rate=self.hrir_sr))
        pass
    
    
    def print_speaker_array(self):
        print('Azimuth cordinates for the speaker array:')
        print(self.ls_az)
        print('Dimensions for azimuth cordinates:')
        print(self.ls_az.shape)
        print('')
        print('Elevation cordinates for the speaker array:')
        print(self.ls_ele)
        print('Dimensions for elevation cordinates:')
        print(self.ls_ele.shape)
        
    def get(self):
        return np.array(self.hrir_array), self.hrir_sr, np.array(self.hrir_vector)
    
    
# VBAP Panner class 
class vbap_panner():
    """Description
    
    Variables
    ----------
    self.hrir_array : np.array, shape=(n, m, 2)
        n = number of speaker array used
        m = length of each HRIR file (one channel)
        2 = number of channels
        HRIR for each speaker position

    self.hrir_sr : int
        Sample rate of HRIR files

    self.hrir_vector : np.array, shape=(n, 2)
        n = number of speaker array used
        2 = azimuth and elevation angle (0 to 360)
        Azimuth and elevation coordinates for each HRIR
        
    self.hrir_vector_cartesion : np.array, shape=(n, 3)
        The [x, y, z] coordinates of each HRIR Vector 
        
    self.triangles : np.arry, shape=(n, 3)
        The indexes of the three speaker vectors in every triangles in the sphere

    Returns
    -------
    vbap_panner class object

    """
    # initialization
    def __init__(self, hrir_array, hrir_sr, hrir_vector):
        # check size
        # if(hrir_array.shape()[0] != hrir_vector.shape()[0]):
          # SHOW ERROR MESSAGE

        self.hrir_array = hrir_array
        self.hrir_sr = hrir_sr
        self.hrir_vector = hrir_vector

        number_of_speakers = len(hrir_vector)

        # create np.array, shape=(number_of_speakers, 3)
        self.hrir_vector_cartesion = np.zeros((number_of_speakers, 3))
        #for each n, compute cartesion fromo angular
        for i in range(number_of_speakers):
            self.hrir_vector_cartesion[i, :] = self.ang_to_cart(hrir_vector[i][0], 
                                                      hrir_vector[i][1])

        # get all triangles around the sphere, and store the speaker's indexes into an array
        self.triangles = ConvexHull(self.hrir_vector_cartesion).simplices
        return

    # spatialize HRIR to correct position in space
    def spatialize(self, sig, sig_sr, azimuth, elevation):

        # force signal to be mono
        # if len(sig.shape)==1:
        #     sig_mono = sig
        # elif len(sig.shape)==2:
        #     sig_mono = np.mean(sig,axis=1)
        # else:
        #     print("Wrong signal dimension.")
        #     return
        
        sig_mono = sig

        if sig_sr != self.hrir_sr:
            sig_mono_resampled = librosa.core.resample(sig_mono,orig_sr=sig_sr,target_sr=self.hrir_sr)
        else:
            sig_mono_resampled = sig_mono

        source_vector_cartesion = self.ang_to_cart(azimuth, elevation)
        gains, ls_index = self.find_active_triangles(source_vector_cartesion)

        # Convolve --> Frequency domain is faster
        L_out = np.zeros(len(sig_mono_resampled)+ self.hrir_array[0].shape[0]-1)
        R_out = np.zeros(len(sig_mono_resampled)+ self.hrir_array[0].shape[0]-1)

        for i in range(3):
          # spatialized source for Left channel
            HRIR_index = ls_index[i]
            L_out +=  signal.fftconvolve(sig_mono_resampled,
                      self.hrir_array[HRIR_index][:, 0].T) * gains[i]
            # spatialized source for Right channel
            R_out += signal.fftconvolve(sig_mono_resampled,
                      self.hrir_array[HRIR_index][:, 1].T) * gains[i]


        Bin_Mix = np.vstack([L_out,R_out]).transpose()
        if np.max(np.abs(Bin_Mix))>0.001:
            Bin_Mix = Bin_Mix/np.max(np.abs(Bin_Mix))
        # print('Data Dimensions: ', Bin_Mix.shape) 
        return Bin_Mix 
    

# Utility Functions:
    # angular to cartesian codinates
    def ang_to_cart(self, azi, ele):
        ele_rad = ele / 360 * 2 * np.pi
        azi_rad = azi / 360 * 2 * np.pi
        x = np.cos(ele_rad) * np.cos(azi_rad)
        y = np.cos(ele_rad) * np.sin(azi_rad)
        z = np.sin(ele_rad)
        return np.array([x, y, z])

    # calculate gain for triangle  
    def calc_gain(self, base, p_cart):
        gains = np.linalg.solve(base, p_cart)
        return gains

    # narmalized gain
    def normalize(self, gains, norm):
        return gains * np.sqrt(norm / np.sum(gains ** 2))

    # find active speaker triangle
    def find_active_triangles(self, p_cart):
        base = np.zeros((3,3))
        for i in range(len(self.triangles)):
            ls_index = self.triangles[i]
            for j in range(3):
                base[:, j] = self.hrir_vector_cartesion[ls_index[j], :]
      
            gains = self.calc_gain(base, p_cart)
            if np.min(gains)>=0:
                gains = self.normalize(gains, 1)
                print("Indexes of speaker array used:"+ str(ls_index))
                break  
        return gains, ls_index     
    
    
def random_generater (num_examples, center_array, width, max_num_tracks,
                      region_seed = 3, angle_seed = 9):
    """Description

    Variables
    ----------
    num_examples : int
        Number of examples

    center_array : np.array, shape=(n,)
        n = number of centers
        Centers'azimuth angles on the elevation 0 plane

    width : int
        Width reletive to centers

    max_num_tracks : int
        Maximum number of tracks in an Ensemble

    Returns
    -------


    """
    array = np.full(shape=(num_examples, 1+max_num_tracks), fill_value=-1, 
                  dtype=int)
    rand_reg = random.Random(region_seed)
    rand_ang = random.Random(angle_seed)

    for i in range(num_examples):
        idx = rand_reg.randint(0, len(center_array)-1)
        array[i][0] = center_array[idx]

        for j in range(1, 1+max_num_tracks):
            array[i][j] = rand_ang.randint(width * -1, width)

    return array


def get_sig_array(source_array, sig_sr=16000, size = 80):
   
    """Description

    Variables
    ----------
    source_array : list, shape(n, num_ins, num_ins_tracks, source_length)
        n = Number of multi-track sources
        num_ins = Number of instruments
        num_ins_tracks = Number of tracks for that instrument
        source_length = duration of the source in second

    sig_sr : int
        Sample rate

    size : int
        Number of examples

    Returns
    -------
    
    sig_array : nd.array, shape(size, 4, sig_sr*duration)

    """
    
    num_sources = len(source_array)
    duration = 3
    
    sig_array = []

    start_seed = 18
    rand_st = random.Random(start_seed)
    
    # Generate sig_array for multitracks
    for i in range(size):
        sig = source_array[i%num_sources]
        
        st = rand_st.randint(30, 120)
        
        sig_truncated = get_sig_multitrack(sig, sig_sr, st, duration)
        sig_array.append(sig_truncated)
    
    return np.array(sig_array)


def get_sig_multitrack(mtrack, sig_sr, st, duration):
    sig_truncated = []
    
    # for each type of instrument
    for ins_idx in range(len(mtrack)):
        ins_truncated = np.zeros(sig_sr*duration)
        
        # for each instrument track
        for track_idx in range(len(mtrack[ins_idx])):
            ins_truncated[:] += mtrack[ins_idx][track_idx][sig_sr*st:sig_sr*(st+duration)]
        
        sig_truncated.append(ins_truncated)
            
    return np.array(sig_truncated)



def generate_data(source_locations, sig_array, sig_sr, panner, size = 500):
    dataset = []

    
    for i in range(size):
        mtrack_example = []
        sig = sig_array[i%len(sig_array)]
        
        
        for j in np.arange(1, len(source_locations[i])):
            azi = source_locations[i][0] + source_locations[i][j]
            ele = 0
            print(str(i) + ":")
            print("Azimuth: " + str(azi) + ", Elevation: 0")
            Bin_Mix = panner.spatialize(sig[j-1], sig_sr, azi, ele)
            # print(Bin_Mix.shape)

            features_L = librosa.feature.melspectrogram(y=Bin_Mix[:, 0], sr=panner.hrir_sr)
            features_R = librosa.feature.melspectrogram(y=Bin_Mix[:, 1], sr=panner.hrir_sr)

            features = np.zeros((features_L.shape[0], features_L.shape[1], 2))
            features[:, :, 0] = features_L
            features[:, :, 1] = features_R
            # print(features_left.shape)

            mtrack_example.append(features)
            
        mtrack_example = np.array(mtrack_example)
            
        dataset.append(np.sum(mtrack_example, axis=0))

    return np.array(dataset)


def draw_sources(source_locations, size=6, colors=['peru','olive', 'lightskyblue', 'purple']):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(1, 1, 1)
    t = np.linspace(0,np.pi*2,100)
    ax.plot(np.cos(t), np.sin(t), linewidth=1)

    
    for i in range(len(source_locations)):
        for j in range(1, len(source_locations[i])):
            azi = source_locations[i][0] + source_locations[i][j]
            w = (azi+90)/180 * np.pi
            ax.plot(np.cos(w), np.sin(w), color=colors[j-1], marker='x')
            
    plt.show()
    
    
def plot_prediction_analysis(y_pred, y_true):
    
    # num_centers, 4 (all, correct, front back, adjacent)
    count = np.zeros((8, 4))
    
    fb = {0:4, 1:3, 2:2, 3:1, 4:0, 5:7, 6:6, 7:5}
    
    for i in range(len(y_pred)):
        count[y_true[i]][0] += 1
        
        if y_pred[i] == y_true[i]:
            count[y_true[i]][1] += 1
        elif np.abs(y_pred[i]-y_true[i]) < 2 or np.abs(y_pred[i] - y_true[i])==7:
            count[y_true[i]][2] += 1
        elif fb[y_pred[i]]==y_true[i]:
            count[y_true[i]][3] +=1
    
    plt.figure(figsize=(18, 3))
    plt.rcParams.update({'font.size': 6})
    labels = ['Count', 'Correct', 'Adjacent', 'Front-Back']
    colors = ['b', 'g', 'y', 'r']
    width=0.15

    x_axis = ['0','45','90','135', '180', '225', '270', '315']

    x = np.arange(8)
    for i in range(4):
        plt.bar(x + (i-1.5)*width, count[:, i], color=colors[i],
                width=width, label=labels[i])
      
    plt.xticks(x, x_axis)
    plt.tight_layout()
    plt.show()
    return count
            
    
import matplotlib.patches as mpatches
def plot_relative_distribution(source_locations, width, size=10, title=''):
    
    
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(1, 1, 1)
    colors = ['salmon', 'gold', 'lightgreen', 'violet']
    labels = ['bass', 'drum', 'guitar', 'piano']
    legend_patch = []

    for i in range(len(colors)):
      legend_patch.append(mpatches.Patch(color=colors[i], label=labels[i]))

    
    # Distribution for all sources
    sources_distribution = [[], [], [], [], [], [], [], []]
    
    for i in range(len(source_locations)):
        for j in range(1, len(source_locations[i])):
            center = source_locations[i][0]
            bias = source_locations[i][j]
            azi = center + bias
            ax.plot(bias, i, color=colors[j-1], marker='x')
    
    ax.set_xlabel('Source Locations Relative to Centers')
    ax.set_ylabel('Example Index')
    fig.legend(handles=legend_patch)
    plt.suptitle(title)
    plt.tight_layout()  
    plt.show()



def plot_polar_distribution(source_locations, width, bins=360, size=(6, 12), title=''):
    
    # Distribution for each instrument
    fig = plt.figure(figsize=size)
    st = 0 / 180 * np.pi
    theta = np.linspace(st, st+2*np.pi, bins)
    source_count = np.zeros((5, bins))

    theta_center = np.linspace(st, st+2*np.pi, 9)[:8]
    source_count_center = np.zeros((5, 8))
    
    plt.rcParams.update({'font.size': 12})

    colors = ['tab:blue', 'salmon', 'gold', 'lightgreen', 'violet']
    labels = ['all', 'bass', 'drum', 'guitar', 'piano']
    
    for i in range(len(source_locations)):
        for j in range(1, len(source_locations[i])):
            center = source_locations[i][0]
            bias = source_locations[i][j]
            azi = center + bias
            
            # center+=90
            if center>=360:
              center-=360

            # azi += 90
            if azi>=360:
                azi -= 360
                
            source_count[0][int(azi/(360/bins))] +=1
            source_count[j][int(azi/(360/bins))] +=1

            source_count_center[0][int(center/45)] +=1
            source_count_center[j][int((center/45))] +=1
                            
              

    legend_patch = []

    for i in range(len(colors)):
      ax1 = plt.subplot(5, 2, i*2+1, polar=True)
      ax1.bar(theta, source_count[i], width=0.01, color=colors[i])
      ax1.set_theta_zero_location("N")

      ax2 = plt.subplot(5, 2, i*2+2, polar=True)
      ax2.bar(theta_center, source_count_center[i], width=width*2/180*np.pi,
              color=colors[i], edgecolor='grey')
      ax2.set_theta_zero_location("N")

      legend_patch.append(mpatches.Patch(color=colors[i], label=labels[i]))
    
    fig.legend(handles=legend_patch)
    plt.rcParams.update({'font.size': 14})
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()