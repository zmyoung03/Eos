#
#   ___ ___  ___                
#  | __/ _ \/ __|               
#  | _| (_) \__ \     
#  |___\___/|___/  E v e n t  D i s p l a y
# _____________________________________________________________________________
#
#  * How to Use
##
#   ___ ___  ___                
#  | __/ _ \/ __|               
#  | _| (_) \__ \     
#  |___\___/|___/  E v e n t  D i s p l a y
# _____________________________________________________________________________
#
#  * How to Use
#
#    Pass in file/folder and plot:
#        EosVis = EosVisualizer("/path/to/file")
#        EosVis.plot_events()
#
#    Options:
#        nEvents [int]    -- # of events to overlay. if not specified,
#                            will plot all events.
#        figpath [str]    -- path to save plot to.
#        useCharge [bool] -- color scaled by charge instead of # of hits.
#                            default is False. not available for hdf5 files.
#
#    Alternatively, plot a single event:
#        EosVis.plot_single_event(event=0, figpath="/where/to/save")
#    If an event # is not specified, it will plot a random event.
#
#    EosVisualizer can take MC ntuples, data ntuples and hdf5 files.
#    - If you pass in a file, it will automatically check what file it is.
#    - If you pass in a directory, you need to specify what type of files
#      in the directory you intend to use:
#        EosVis = EosVisualizer("/path/to/folder", filetype="hdf5") # or "root"
#
#    You can also feed in data manually instead of reading it from a file:
#        EosVis = EosVisualizer() # do not specify filepath
#        EosVis.load_hits(pmt_hits) # pmt_hits is a dictionary with LCNs as keys
#                                     and nhits as values
#    Or,
#        EosVis.pmt_hits[lcn] = nhit # assign numbers directly
#
#
#  * Formatting options
#
#     EosViz.show_lcn         -- default:False. Shows LCNs next to PMTs.
#     EosViz.show_colorbar    -- default:True. Shows colorbar(scale) on plot.
#     EosViz.mark_dichroicons -- default:True. Marks dichroicons with yellow halo.
#     EosViz.disable_channels(list_of_channels)
#         -- pass in a list of LCNs or PMT IDs to show as dark grey.
#     EosViz.add_marker(x, y, z, marker="*", color="magenta", size=2000)
#         -- add markers to event display at given positions.
#
# to use: cd into the folder this is stored in, then do: python3 EventDisplay3D.py /(filepath)/(root file).root

'''----------------IMPORTS----------------'''

import os
import re
import math
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import h5py
import ROOT
import PMT_POSITIONS
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3D

'''----------------GEOMETRY----------------'''

def arrow3d(ax, start=(-502.0, 870.4, 571.5), direction_ar=(0, 1, 0), length=1000, width=500, head=0.2, headwidth=1.5, theta_x=0, theta_y=0, theta_z=0, text_tip=None, text_base=None, tip_offset=(0, 0, 0), base_offset=(0, 0, 0), **kw):
    
   '''Draw a 3D arrow on a 3D axis using plot_surface.
   :param ax: The 3D axis to draw the arrow on
   :param start: Starting point of the arrow
   :param direction_ar: Direction vector the arrow points to
   :param length: Length of the arrow
   :param width: Width of the arrow's shaft
   :param head: Fraction of the arrow that forms the head
   :param headwidth: Width of the arrow's head
   :param theta_x: Rotation around the x-axis in degrees
   :param theta_y: Rotation around the y-axis in degrees
   :param theta_z: Rotation around the z-axis in degrees
   :param text: Text label for the arrow
   :param text_offset: Offset for the text label position
   :param kw: Additional keyword arguments for plot_surface'''

   # Normalize direction vector to prevent scaling issues
   direction_ar = np.array(direction_ar)
   if np.linalg.norm(direction_ar) == 0:
      direction_ar = np.array([0, 1, 0]) # Default direction
   else:
      direction_ar = direction_ar / np.linalg.norm(direction_ar) # Normalize direction vector
   #print("Normalized direction:", direction)
   
   # Scale the direction vector by the length of the arrow
   direction_ar *= length
   
   # Define the arrow body and head
   a_body = np.array([[0, 0], [width, 0], [width, (1 - head) * length], [0, (1 - head) * length]])
   a_head = np.array([[0, (1 - head) * length], [headwidth * width, (1 - head) * length], [0, length]])

   # Create the arrow by revolving around the z-axis
   r_body, theta_body = np.meshgrid(a_body[:, 0], np.linspace(0, 2 * np.pi, 30))
   r_head, theta_head = np.meshgrid(a_head[:, 0], np.linspace(0, 2 * np.pi, 30))
   z_body = np.tile(a_body[:, 1], r_body.shape[0]).reshape(r_body.shape)
   z_head = np.tile(a_head[:, 1], r_head.shape[0]).reshape(r_head.shape)
   x_body = r_body * np.sin(theta_body)
   y_body = r_body * np.cos(theta_body)
   x_head = r_head * np.sin(theta_head)
   y_head = r_head * np.cos(theta_head)
   
   # Rotation matrices for x, y, and z axes
   rot_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
   rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
   rot_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

   # Apply rotations and translate by the start point
   b1_body = np.dot(rot_x, np.dot(rot_y, np.dot(rot_z, np.c_[x_body.flatten(), y_body.flatten(), z_body.flatten()].T)))
   b2_body = b1_body.T + np.array(start)

   b1_head = np.dot(rot_x, np.dot(rot_y, np.dot(rot_z, np.c_[x_head.flatten(), y_head.flatten(), z_head.flatten()].T)))
   b2_head = b1_head.T + np.array(start)
   
   x_body = b2_body[:, 0].reshape(r_body.shape)
   y_body = b2_body[:, 1].reshape(r_body.shape)
   z_body = b2_body[:, 2].reshape(r_body.shape)
   
   x_head = b2_head[:, 0].reshape(r_head.shape)
   y_head = b2_head[:, 1].reshape(r_head.shape)
   z_head = b2_head[:, 2].reshape(r_head.shape)
   
   ax.plot_surface(x_body, y_body, z_body, **kw)
   ax.plot_surface(x_head, y_head, z_head, **kw)
   
   # Calculate the tip position of the arrow
   tip_position = np.array(start) + direction_ar
   tip_text_position = tip_position + np.array(tip_offset)
   base_text_position = np.array(start) + np.array(base_offset)

   # Add text at the tip if provided
   if text_tip:
      ax.text(tip_text_position[0], tip_text_position[1], tip_text_position[2], text_tip, fontsize=16, color=kw.get('color', 'black'))
      
# Add text at the base if provided
   if text_base:
      ax.text(base_text_position[0], base_text_position[1], base_text_position[2], text_base, fontsize=16, color=kw.get('color', 'black'))
      
      
      
     
def cone3d(ax, start=(-502.0, 870.4, 571.5), direction_cn=(0, 1, 0), length=1000, radius=500, theta_x=0, theta_y=0, theta_z=0, text_tip=None, text_base=None, tip_offset=(0, 0, 0), base_offset=(0, 0, 0), **kw):
    '''Draw a 3D cone on a 3D axis using plot_surface.
    :param ax: The 3D axis to draw the cone on
    :param start: Starting point of the cone
    :param direction_cn: Direction vector the cone points to
    :param length: Length of the cone
    :param radius: Base radius of the cone
    :param theta_x: Rotation around the x-axis in degrees
    :param theta_y: Rotation around the y-axis in degrees
    :param theta_z: Rotation around the z-axis in degrees
    :param text_tip: Text label for the cone tip
    :param text_base: Text label for the cone base
    :param tip_offset: Offset for the tip text label position
    :param base_offset: Offset for the base text label position
    :param kw: Additional keyword arguments for plot_surface'''

    # Normalize direction vector to prevent scaling issues
    direction_cn = np.array(direction_cn)
    if np.linalg.norm(direction_cn) == 0:
        direction_cn = np.array([0, 1, 0])  # Default direction
    else:
        direction_cn = direction_cn / np.linalg.norm(direction_cn)  # Normalize direction vector

    # Scale the direction vector by the length of the cone
    direction_cn *= length
    
    # Define the tip of the cone
    tip = np.array(start)
    
    #Calculate the base position
    base_position = tip - direction_cn

    # Create a circular base for the cone
    num_points = 30
    angles = np.linspace(0, 2 * np.pi, num_points)
    x_base = radius * np.cos(angles)
    y_base = radius * np.sin(angles)
    z_base = np.zeros_like(x_base)

    # Create the cone surface by connecting the base to the tip
    X, Y, Z = [], [], []
    for i in range(num_points):
        X.append([x_base[i], 0])  # Base point and tip
        Y.append([y_base[i], 0])  # Base point and tip
        Z.append([z_base[i], length])  # Base level and tip height
    
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # Rotation matrices for x, y, and z axes
    rot_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rot_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

    # Apply rotations and translate by the start point
    base_points = np.array([x_base, y_base, z_base]).T
    tip_point = np.array([0, 0, length])
    
    rotated_base = np.dot(rot_x, np.dot(rot_y, np.dot(rot_z, base_points.T))).T + base_points
    rotated_tip = np.dot(rot_x, np.dot(rot_y, np.dot(rot_z, tip_point))) + np.array(start)

    # Plot the cone surface
    for i in range(num_points):
        ax.plot([rotated_base[i][0], rotated_tip[0]], [rotated_base[i][1], rotated_tip[1]], [rotated_base[i][2], rotated_tip[2]], **kw)

    # Add text at the tip if provided
    if text_tip:
        tip_text_position = rotated_tip + np.array(tip_offset)
        ax.text(tip_text_position[0], tip_text_position[1], tip_text_position[2], text_tip, fontsize=16, color=kw.get('color', 'black'))

    # Add text at the base if provided
    if text_base:
        base_text_position = np.array(start) + np.array(base_offset)
        ax.text(base_text_position[0], base_text_position[1], base_text_position[2], text_base, fontsize=16, color=kw.get('color', 'black'))

'''----------------DEFINITIONS----------------'''

def naturalSort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    natsort_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=natsort_key)

def getFilesUnderFolder(path, filetype=""):
    """
    get list of files under folder.
    will only find files of filetype if specified --
    pass in string without leading "." (ex. "root", "pkl")
    """
    if filetype:
        l = len(filetype)
        return [os.path.join(path,f) for f in os.listdir(path)
                if os.path.isfile(os.path.join(path,f))
                and f[-(l+1):] == f".{filetype}"]
    else:
        return [os.path.join(path,f) for f in os.listdir(path)
                if os.path.isfile(os.path.join(path,f))]

def getFileName(path):
    '''
    Get name of file, without file extension or parent folder
    '''
    namepieces = path.split("/")[-1].split(".")
    filename = ".".join(namepieces[:-1])
    return filename

def getFolderName(path):
    """
    find name of lowest level directory of given path.
    """
    if os.path.isfile(path):
        path = os.path.dirname(path)
    return os.path.basename(os.path.normpath(path))

def xy_phi(x, y):
    """
    map x, y coordinates to phi.
    phi == 0 on +y axis, increasing CCW when viewing from above.
    """
    xsign = np.where(np.sign(x) == 0, 1, np.sign(x))
    phi = np.where((x==0)&(y==0), 0, xsign*np.arccos(y/np.sqrt(x**2+y**2)))
    return phi

def rotate_pmtarray(points, angle, origin=(0,0)):
    """
    rotate points CCW in the xy-plane by a given angle around a given origin.
    angle should be given in radians.
    """
    ox, oy = origin
    rpos = []
    for point in points:
        px, py, pz = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        rpos.append([qx, qy, pz])
    rpos = np.vstack(rpos)
    return rpos

def modify_cmap(original_cmap, maxval):
    """
    make viridis cmap where only -1 ~ 0 is grey
    """
    viridis = plt.get_cmap(original_cmap, maxval)
    newcolor = viridis(np.linspace(0, 1, maxval+1))
    grey = np.array([32/256, 32/256, 32/256, 1])
    newcolor[:1, :] = grey
    return matplotlib.colors.ListedColormap(newcolor)

class EosVisualizer():
    """Class for Visualizing Eos Events"""

    def __init__(self, filepath=None, filetype=None):
        """ Initialize for different types of inputs """
        
        # if no filepath was given, assume data will be given manually
        if filepath == None:
            self.init_pmt_positions()
            self.filetype = "manual"

        # filepath is one file, check for file type
        elif filepath.split(".")[-1] == "h5":
            self.init_hdf5(filepath)

        elif filepath.split(".")[-1] == "root":
            self.init_root(filepath)

        # filepath is directory,
        # need user to manually specify file type to parse for.
        elif os.path.isdir(filepath):
            if filetype == None:
                raise Exception("Please specify filetype (hdf5 / root).")
            elif filetype == "hdf5":
                self.init_hdf5_multiple(filepath)
            elif filetype == "root":
                self.init_root_multiple(filepath)

        # settings
        self.threshold = -5.0 # threshold to pass to count as hit, in mV
        self.termination_ohms = 50
        self.boards = PMT_POSITIONS.boards
        self.ped_window = 30 # samples

        # initiate default for labels and markers
        self.show_lcn = False
        self.show_colorbar = True
        self.mark_dichroicons = True
        self.disabled_channels = []
        self.markers = []

        self.lcn_fontsize = 12

        dic_lcn = np.arange(7*16, 7*16+12)
        if "mc" in self.filetype:
            self.dichroicons = [self.pmtid2lcn.index(i) for i in dic_lcn]
        else:
            self.dichroicons = dic_lcn
    
    def init_pmt_positions(self):
        """ Initialize PMT positions from PMT_POSITIONS.py """

        pmtx = PMT_POSITIONS.x
        pmty = PMT_POSITIONS.y
        pmtz = PMT_POSITIONS.z
        self.init_pmt_hits(pmtx, pmty, pmtz)
    
    def init_pmt_hits(self, pmtx, pmty, pmtz):
        """ Initiate dict to store pmt hits """

        # global rotation to let top PMT array lie horizontally on EventDisplay
        self.pmtpos = rotate_pmtarray(np.stack((pmtx, pmty, pmtz), axis=1), np.pi/2.)

        # initiate dictionary to store charges
        self.pmt_hits = {}
        for lcn in range(len(self.pmtpos)):
            if self.pmtpos[lcn][2] == -1: # not a PMT
                continue
            self.pmt_hits[lcn] = 0
        
        self.LCNs = list(self.pmt_hits.keys())

    def init_hdf5(self, filepath):
        self.init_pmt_positions()
        self.file = h5py.File(filepath, "r")
        b0 = list(self.file.keys())[0]
        ch0 = [key for key in self.file[b0].keys() if "ch" in key][0]
        self.nEvents = len(self.file[b0][ch0]['samples'])
        self.filetype = "hdf5"
    
    def init_hdf5_multiple(self, filepath):
        self.init_pmt_positions()
        self.file = naturalSort(getFilesUnderFolder(filepath, "h5"))
        self.nFiles = len(self.file)
        self.filetype = "hdf5_multiple"

    def init_root(self, filepath):
        self.file = ROOT.TFile.Open(filepath)
        meta = self.file.Get("meta")
        meta.GetEntry(0)

        try:
            self.tree = self.file.Get("output")
            self.filetype = "mc"
            pmtx = list(meta.pmtX)
            pmty = list(meta.pmtY)
            pmtz = [z-170 for z in list(meta.pmtZ)]
            self.pmtid2lcn = list(meta.pmtChannel)
        except:
            self.tree = self.file.Get("events")
            self.filetype = "data"
            pmtx = list(meta.pmtx)
            pmty = list(meta.pmty)
            pmtz = list(meta.pmtz)

        self.nEvents = self.tree.GetEntries()
        self.init_pmt_hits(pmtx, pmty, pmtz)

    def init_root_multiple(self, filepath):
        self.file = naturalSort(getFilesUnderFolder(filepath, "root"))
        self.nFiles = len(self.file)
        f0 = ROOT.TFile.Open(self.file[0])
        meta = f0.Get("meta")
        meta.GetEntry(0)

        try:
            tree = f0.Get("output")
            self.filetype = "mc_multiple"
            pmtx = list(meta.pmtX)
            pmty = list(meta.pmtY)
            pmtz = [z-170 for z in list(meta.pmtZ)]
            self.pmtid2lcn = list(meta.pmtChannel)
        except:
            tree = f0.Get("events")
            self.filetype = "data_multiple"
            pmtx = list(meta.pmtx)
            pmty = list(meta.pmty)
            pmtz = list(meta.pmtz)
            
        self.init_pmt_hits(pmtx, pmty, pmtz)

    def load_hits_from_file(self, events, useCharge=False):
        """
        load hit/charge information from files for given list of events.
        any event # that exceeds total number of events will be ignored.
        """
        if self.filetype == "hdf5":
            self.load_from_hdf5(self.file, events)
        elif self.filetype == "mc":
            self.load_from_mc(self.tree, events, useCharge=useCharge)
        elif self.filetype == "data":
            self.load_from_data(self.tree, events, useCharge=useCharge)
        elif "multiple" in self.filetype:
            events_left, offset = events, 0
            for filepath in self.file:
                if len(events_left) == 0:
                    break
                if self.filetype == "hdf5_multiple":
                    with h5py.File(filepath, "r") as f:
                        nEvents = self.load_from_hdf5(f, events_left)
                if self.filetype == "mc_multiple":
                    f = ROOT.TFile.Open(filepath)
                    tree = f.Get("output")
                    nEvents = self.load_from_mc(tree, events_left, useCharge=useCharge)
                    f.Close()
                if self.filetype == "data_multiple":
                    f = ROOT.TFile.Open(filepath)
                    tree = f.Get("events")
                    nEvents = self.load_from_data(f, events_left, useCharge=useCharge)
                    f.Close()
                events_left = [ev-nEvents for ev in events_left if ev>=nEvents]
                offset += nEvents

    def load_from_hdf5(self, f, events):
        for b in list(f.keys()):
            b_id = self.boards[b]
            bits = f[b].attrs['bits']
            channels = [ch for ch in list(f[b].keys()) if ch[:2] == "ch"] 

            for ch in channels:
                ch_num = int(ch.replace("ch",""))
                lcn = ch_num + b_id*16
                if not lcn in self.LCNs:
                    continue

                dynamic_range = f[b][ch].attrs['dynamic_range']
                dv = dynamic_range/np.power(2, bits)
                data = np.array(f[b][ch]["samples"], dtype=np.float32)
                nEvents = len(data)

                for ev in events:
                    if ev >= nEvents:
                        break
                    pedestal = np.mean(data[ev][0:self.ped_window])
                    voltage = ((data[ev])-pedestal) * dv
                    if any(voltage < self.threshold):
                        self.pmt_hits[lcn] += 1
        return nEvents
    
    def load_from_mc(self, tree, events, useCharge=False):
        nEvents = tree.GetEntries()
        for ev in events:
            if ev >= nEvents:
                break
            tree.GetEntry(ev)
            hitPMTs = list(tree.hitPMTID)
            if useCharge:
                charges = list(tree.hitPMTCharge)
        
            for (i, pmtid) in enumerate(hitPMTs):
                if useCharge:
                    self.pmt_hits[pmtid] += charges[i]
                else:
                    self.pmt_hits[pmtid] += 1
        return nEvents
    
    def load_from_data(self, tree, events, useCharge=False):
        nEvents = tree.GetEntries()
        for ev in events:
            if ev >= nEvents:
                break
            tree.GetEntry(ev)
            hitLCNs = list(tree.lcn)
            if useCharge:
                charges = list(tree.charge)
        
            for (i, lcn) in enumerate(hitLCNs):
                if useCharge:
                    self.pmt_hits[lcn] += charges[i]
                else:
                    self.pmt_hits[lcn] += 1
        return nEvents

    def sorthits(self):
        """
        sort hits in self.pmt_hits into dictionary
        based on position in Eos (top, side, bottom lo and hi)
        """
        hits    = {"top":[], "side":[], "dic_hi":[], "dic_lo":[]}
        charges = {"top":[], "side":[], "dic_hi":[], "dic_lo":[]}

        # PMT Arrays are delineated by PMT z-position
        for lcn, q in self.pmt_hits.items():
            pos = self.pmtpos[lcn]
            if lcn in self.disabled_channels:
                q = -1 # negative value will be marked as grey
            if pos[2] > 900:
                hits["top"].append(pos)
                charges["top"].append(q)
            elif pos[2] > -700 and pos[2] < 900:
                hits["side"].append(pos)
                charges["side"].append(q)
            elif pos[2] < -700 and pos[2] > -1300:
                hits["dic_hi"].append(pos)
                charges["dic_hi"].append(q)
            elif pos[2] < -1300:
                hits["dic_lo"].append(pos)
                charges["dic_lo"].append(q)

        for loc in hits.keys():
            hits[loc] = np.array(hits[loc])
            charges[loc] = np.array(charges[loc])

        return hits, charges

    def clear_hits(self):
        """clear pmt hit charge dict after plotting"""
        for hit, q in self.pmt_hits.items():
            self.pmt_hits[hit] = 0
    
    def load_hits(self, pmt_hits):
        """
        manually feed in hits.
        pmt_hits should be a dict with LCNs as keys and nhits as values.
        """
        for lcn, q in self.pmt_hits.items():
            try:
                self.pmt_hits[lcn] = q
            except:
                pass
    
    def disable_channels(self, channels):
        """ pass in list of channels to disable """
        for ch in channels:
            if not ch in self.disabled_channels:
                self.disabled_channels.append(ch)
    
    def add_marker(self, x, y, z, marker="*", color="magenta", size=2000):
        """ add marker to event display """
        m = {'pos':(x,y,z), 'm':marker, 'c':color, 's':size}
        self.markers.append(m)

    def plot_events(self, nEvents=-1, figpath=None, useCharge=False):
        """
        Display multiple Eos events overlayed
        """

        if not self.filetype == "manual":
            if nEvents < 0:
                if "multiple" in self.filetype:
                    maxEv = 1000000
                else:
                    maxEv = self.nEvents
            else:
                maxEv = nEvents
            self.load_hits_from_file(np.arange(0,maxEv,1), useCharge=useCharge)

        hits, charges = self.sorthits()
        self.plot(hits, charges, figpath)
        self.clear_hits()
    
    def plot_single_event(self, event=-1, figpath=None, useCharge=False):
        """
        Display single Eos event
        """
        if event < 0:
            try:
                events = [np.random.randint(self.nEvents)]
            except:
                events = [np.random.randint(self.nFiles)]
        else:
            events = [event]
        self.load_hits_from_file(events, useCharge=useCharge)

        hits, charges = self.sorthits()
        self.plot(hits, charges, figpath)
        self.clear_hits()

    def plot(self, hits, charges, figpath=None):
        # Calculate radius for side PMT display aspect
        r = np.average(np.sqrt(hits["side"][:,0]**2+hits["side"][:,1]**2))

        # pmt marker size
        pmtsize = 200

       # Find highest charge pmt for colorbar
        all_charges = np.concatenate([charges["top"], charges["side"], charges["dic_hi"], charges["dic_lo"]])
        maxval = max(all_charges)
        if self.disabled_channels:
            norm = matplotlib.colors.Normalize(vmin=-1, vmax=maxval)
            cmap = modify_cmap('viridis', maxval)
        else:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=maxval)
            cmap = plt.get_cmap('viridis')

        # Display Canvas and axes
        fig = plt.figure(figsize=(18, 14), facecolor='black')
        gs = fig.add_gridspec(3, 2) # GridSpec for original layout
        
        # Create 3D axes
        ax = fig.add_subplot(111, projection='3d')
        
        # Set view angle:
        ax.view_init(vertical_axis = "y")  
        #vertical_axis = "y"
         #At the moment the side hits and top/bottom hits can't be plotted on the same graph and make a coherent shape. That's something else I'm working on fixing. If you comment out paragraphs under the "plot x hit" comment, they'll work individually
        
        # Plot side hits
        if len(hits["side"]) > 0:
               # Calculate radius and angle
               phi = xy_phi(hits["side"][:,0], hits["side"][:,1])
               radius = np.average(np.sqrt(hits["side"][:,0]**2 + hits["side"][:,1]**2))
    
               # Convert polar coordinates to Cartesian coordinates for a curved surface
               r = 4 * radius
               x_pos = r * np.cos(phi)
               y_pos = r * np.sin(phi)
    
               # Plot the side PMTs
               ax.scatter(x_pos, hits["side"][:,2], y_pos, 
                      s=pmtsize, c=charges["side"], norm=norm, cmap=cmap, label='Side Hits')
               #EosViz.add_marker(x=-502.0, y=870.4, z=571.5, marker="*", color="red", size=200)
        
        '''# Plot top hits
        if len(hits["top"]) > 0:
            ax.scatter(hits["top"][:,0], hits["top"][:,1], hits["top"][:,2],
                   s=pmtsize, c=charges["top"], norm=norm, cmap=cmap, label='Top Hits')    
                   
        # Plot dic_hi hits
        if len(hits["dic_hi"]) > 0:
            ax.scatter(hits["dic_hi"][:,0], hits["dic_hi"][:,1], hits["dic_hi"][:,2], 
                   s=pmtsize, c=charges["dic_hi"], norm=norm, cmap=cmap, label='DIC HI Hits')

        #np.ones(len(hits["dic_hi"]))*1000, np.ones(len(hits["dic_lo"]))*-1000,(goes between [:,1] and s=pmtsize when [:,2] is removed
        #this is a note to myself I can explain it if it doesn't make sense
        
        # Plot dic_lo hits
        if len(hits["dic_lo"]) > 0:
            ax.scatter(hits["dic_lo"][:,0], hits["dic_lo"][:,1], hits["dic_lo"][:,2], 
                   s=pmtsize, c=charges["dic_lo"], norm=norm, cmap=cmap, label='DIC LO Hits')'''

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Insert colorbar
        if self.show_colorbar:
            cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.8])
            cb = fig.colorbar(ax.collections[0], cax=cbar_ax)
            cbar_ax.set_ylim(bottom=0)
            fg_color = "white"
            cb.ax.yaxis.set_tick_params(color=fg_color, labelsize=24)
            cb.outline.set_edgecolor(fg_color)
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)

        '''# Mark dichroicons
        if self.mark_dichroicons:
            dic = np.array([self.pmtpos[lcn] for lcn in self.dichroicons])
            ax.scatter(dic[:,0], dic[:,1], dic[:,2], s=pmtsize*2, c="black",
                   edgecolor="goldenrod", linewidth=3, zorder=-1, label='Dichroicons')'''

        # Label LCNs
        if self.show_lcn:
            labeloffset = 60
            for pmtid in self.LCNs:
                pmtpos = self.pmtpos[pmtid]
                ax.text(pmtpos[0], pmtpos[1], pmtpos[2], str(pmtid), color="w", size=self.lcn_fontsize, ha="center")

        # Draw markers
        for m in self.markers:
            mpos = rotate_pmtarray([m['pos']], np.pi/2.)[0]
            ax.scatter(mpos[0], mpos[1], mpos[2], marker=m['m'], c=m['c'], s=m['s'], label='Markers')

        # Add legend
        # ax.legend()

        # Save and show
        if figpath:
            plt.savefig(figpath, dpi=300)
            print(f"Plot saved to {figpath}")
        else:
            plt.show()
            
        # Overlay a 3D arrow without blocking the data
        
        arrow3d(
            ax, # ax_3d: the 3D axis object where the arrow will be plotted.
            start=[-502.0, 870.4, 571.5], # start: the starting point coordinates (x, y, z) of the arrow.
            direction_ar=[0, 1, 0], # direction: vector indicating the direction of the arrow.
            length=500, # length: length of the arrow from base to tip.
            width=50, # width: width of the arrow's shaft.
            head=0.4, # head: proportion of the total length that the head occupies.
            headwidth=2, # headwidth: multiplier that determines the width of the arrow's head relative to the shaft.
            theta_x=38, # theta_x: rotation angle around the x-axis in degrees. 90 = in -y directuion. -90 = in y direction.
            theta_y=0, # theta_y: rotation angle around the y-axis in degrees. -90 = in -x direction. 90 = in positive x direction
            theta_z=0, # theta_z: rotation angle around the z-axis in degrees.
            #(0, 0, 0) is along +z axis. 
            color='green', # color: color of the arrow.
            text_tip="", # text_tip: text label placed near the tip of the arrow.
            text_base="", # text_base: text label placed near the base of the arrow.
            tip_offset=(-7500, 3000, 0), # tip_offset: moves the tip text relative to the tip's position.
            base_offset=(1000, 0, 0), # base_offset: moves the base text relative to the base's position.
            alpha=0.5 # alpha: transparency of the arrow, where 1 is opaque and 0 is fully transparent.
            )

        # Save and show the plot
        if figpath:
           plt.savefig(figpath, dpi=300)
           print(f"Plot saved to {figpath}")
        else:
           plt.show()
        
        cone3d(
            ax,  # ax: the 3D axis object where the cone will be plotted.
            start=[-502.0, 870.4, 571.5],  # start: the starting point coordinates (x, y, z) of the cone.
            direction_cn=[0, 0, 1],  # direction: vector indicating the direction of the cone.
            length=5000,  # length: length of the cone from base to tip.
            radius=500,  # radius: base radius of the cone.
            theta_x=38,  # theta_x: rotation angle around the x-axis in degrees.
            theta_y=0,  # theta_y: rotation angle around the y-axis in degrees.
            theta_z=0,  # theta_z: rotation angle around the z-axis in degrees.
            color='blue',  # color: color of the cone.
            text_tip="",  # text_tip: text label placed near the tip of the cone.
            text_base="",  # text_base: text label placed near the base of the cone.
            tip_offset=(-7500, 3000, 0),  # tip_offset: moves the tip text relative to the tip's position.
            base_offset=(1000, 0, 0),  # base_offset: moves the base text relative to the base's position.
            alpha=0.5  # alpha: transparency of the cone, where 1 is opaque and 0 is fully transparent.
            )
    
        # Save and show the plot
        if figpath:
           plt.savefig(figpath, dpi=300)
           print(f"Plot saved to {figpath}")
        else:
           plt.show()
           
'''----------------IF STATEMENTS + RUN PLOT (?)----------------'''       

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("f", action="store", help="pass in ntuple file to plot")
    parser.add_argument("-s", "--single", action="store_true", help="plot event display of single event. default is to plot multiple")
    parser.add_argument("-c", "--useCharge", type=bool, default=True, help="use charge? uses nhit if set to False")
    parser.add_argument("-e", "--events", type=int, default=-1, help="number of events to be overlayed, when plotting multiple events")
    args = parser.parse_args()
    
    plotdir = "./EventDisplays/"
    try: os.makedirs(plotdir)
    except: pass

    EosViz = EosVisualizer(args.f)
    filename = getFileName(args.f)
    figpath = os.path.join(plotdir, filename+".png")

    if args.single:
        EosViz.plot_single_event(useCharge=args.useCharge, figpath=figpath)
    else:
        EosViz.plot_events(nEvents=args.events, useCharge=args.useCharge, figpath=figpath)
        
EosVis = EosVisualizer("/path/to/file")
EosVis.plot_single_event(event=0, figpath="")
#    Pass in file/folder and plot:
#        EosVis = EosVisualizer("/path/to/file")
#        EosVis.plot_events()
#
#    Options:
#        nEvents [int]    -- # of events to overlay. if not specified,
#                            will plot all events.
#        figpath [str]    -- path to save plot to.
#        useCharge [bool] -- color scaled by charge instead of # of hits.
#                            default is False. not available for hdf5 files.
#
#    Alternatively, plot a single event:
#        EosVis.plot_single_event(event=0, figpath="/where/to/save")
#    If an event # is not specified, it will plot a random event.
#
#    EosVisualizer can take MC ntuples, data ntuples and hdf5 files.
#    - If you pass in a file, it will automatically check what file it is.
#    - If you pass in a directory, you need to specify what type of files
#      in the directory you intend to use:
#        EosVis = EosVisualizer("/path/to/folder", filetype="hdf5") # or "root"
#
#    You can also feed in data manually instead of reading it from a file:
#        EosVis = EosVisualizer() # do not specify filepath
#        EosVis.load_hits(pmt_hits) # pmt_hits is a dictionary with LCNs as keys
#                                     and nhits as values
#    Or,
#        EosVis.pmt_hits[lcn] = nhit # assign numbers directly
#
#
#  * Formatting options
#
#     EosViz.show_lcn         -- default:False. Shows LCNs next to PMTs.
#     EosViz.show_colorbar    -- default:True. Shows colorbar(scale) on plot.
#     EosViz.mark_dichroicons -- default:True. Marks dichroicons with yellow halo.
#     EosViz.disable_channels(list_of_channels)
#         -- pass in a list of LCNs or PMT IDs to show as dark grey.
#     EosViz.add_marker(x, y, z, marker="*", color="magenta", size=2000)
#         -- add markers to event display at given positions.
#
# to use: cd into the folder this is stored in, then do: python3 EventDisplay3D.py /(filepath)/(root file).root

'''----------------IMPORTS----------------'''

import os
import re
import math
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import h5py
import ROOT
import PMT_POSITIONS
from mpl_toolkits.mplot3d import Axes3D

'''----------------GEOMETRY----------------'''

def arrow3d(ax, start=(-502.0, 870.4, 571.5), direction_ar=(0, 1, 0), length=1000, width=500, head=0.2, headwidth=1.5, theta_x=0, theta_y=0, theta_z=0, text_tip=None, text_base=None, tip_offset=(0, 0, 0), base_offset=(0, 0, 0), **kw):
    
   '''Draw a 3D arrow on a 3D axis using plot_surface.
   :param ax: The 3D axis to draw the arrow on
   :param start: Starting point of the arrow
   :param direction_ar: Direction vector the arrow points to
   :param length: Length of the arrow
   :param width: Width of the arrow's shaft
   :param head: Fraction of the arrow that forms the head
   :param headwidth: Width of the arrow's head
   :param theta_x: Rotation around the x-axis in degrees
   :param theta_y: Rotation around the y-axis in degrees
   :param theta_z: Rotation around the z-axis in degrees
   :param text: Text label for the arrow
   :param text_offset: Offset for the text label position
   :param kw: Additional keyword arguments for plot_surface'''

   # Normalize direction vector to prevent scaling issues
   direction_ar = np.array(direction_ar)
   if np.linalg.norm(direction_ar) == 0:
      direction_ar = np.array([0, 1, 0]) # Default direction
   else:
      direction_ar = direction_ar / np.linalg.norm(direction_ar) # Normalize direction vector
   #print("Normalized direction:", direction)
   
   # Scale the direction vector by the length of the arrow
   direction_ar *= length
   
   # Define the arrow body and head
   a_body = np.array([[0, 0], [width, 0], [width, (1 - head) * length], [0, (1 - head) * length]])
   a_head = np.array([[0, (1 - head) * length], [headwidth * width, (1 - head) * length], [0, length]])

   # Create the arrow by revolving around the z-axis
   r_body, theta_body = np.meshgrid(a_body[:, 0], np.linspace(0, 2 * np.pi, 30))
   r_head, theta_head = np.meshgrid(a_head[:, 0], np.linspace(0, 2 * np.pi, 30))
   z_body = np.tile(a_body[:, 1], r_body.shape[0]).reshape(r_body.shape)
   z_head = np.tile(a_head[:, 1], r_head.shape[0]).reshape(r_head.shape)
   x_body = r_body * np.sin(theta_body)
   y_body = r_body * np.cos(theta_body)
   x_head = r_head * np.sin(theta_head)
   y_head = r_head * np.cos(theta_head)
   
   # Rotation matrices for x, y, and z axes
   rot_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
   rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
   rot_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

   # Apply rotations and translate by the start point
   b1_body = np.dot(rot_x, np.dot(rot_y, np.dot(rot_z, np.c_[x_body.flatten(), y_body.flatten(), z_body.flatten()].T)))
   b2_body = b1_body.T + np.array(start)

   b1_head = np.dot(rot_x, np.dot(rot_y, np.dot(rot_z, np.c_[x_head.flatten(), y_head.flatten(), z_head.flatten()].T)))
   b2_head = b1_head.T + np.array(start)
   
   x_body = b2_body[:, 0].reshape(r_body.shape)
   y_body = b2_body[:, 1].reshape(r_body.shape)
   z_body = b2_body[:, 2].reshape(r_body.shape)
   
   x_head = b2_head[:, 0].reshape(r_head.shape)
   y_head = b2_head[:, 1].reshape(r_head.shape)
   z_head = b2_head[:, 2].reshape(r_head.shape)
   
   ax.plot_surface(x_body, y_body, z_body, **kw)
   ax.plot_surface(x_head, y_head, z_head, **kw)
   
   # Calculate the tip position of the arrow
   tip_position = np.array(start) + direction_ar
   tip_text_position = tip_position + np.array(tip_offset)
   base_text_position = np.array(start) + np.array(base_offset)

   # Add text at the tip if provided
   if text_tip:
      ax.text(tip_text_position[0], tip_text_position[1], tip_text_position[2], text_tip, fontsize=16, color=kw.get('color', 'black'))
      
# Add text at the base if provided
   if text_base:
      ax.text(base_text_position[0], base_text_position[1], base_text_position[2], text_base, fontsize=16, color=kw.get('color', 'black'))
      
      
      
     
def cone3d(ax, start=(-502.0, 870.4, 571.5), direction_cn=(0, 1, 0), length=1000, radius=500, theta_x=0, theta_y=0, theta_z=0, text_tip=None, text_base=None, tip_offset=(0, 0, 0), base_offset=(0, 0, 0), **kw):
    '''Draw a 3D cone on a 3D axis using plot_surface.
    :param ax: The 3D axis to draw the cone on
    :param start: Starting point of the cone
    :param direction_cn: Direction vector the cone points to
    :param length: Length of the cone
    :param radius: Base radius of the cone
    :param theta_x: Rotation around the x-axis in degrees
    :param theta_y: Rotation around the y-axis in degrees
    :param theta_z: Rotation around the z-axis in degrees
    :param text_tip: Text label for the cone tip
    :param text_base: Text label for the cone base
    :param tip_offset: Offset for the tip text label position
    :param base_offset: Offset for the base text label position
    :param kw: Additional keyword arguments for plot_surface'''

    # Normalize direction vector to prevent scaling issues
    direction_cn = np.array(direction_cn)
    if np.linalg.norm(direction_cn) == 0:
        direction_cn = np.array([0, 1, 0])  # Default direction
    else:
        direction_cn = direction_cn / np.linalg.norm(direction_cn)  # Normalize direction vector

    # Scale the direction vector by the length of the cone
    direction_cn *= length
    
    # Define the tip of the cone
    tip = np.array(start)
    
    #Calculate the base position
    base_position = tip - direction_cn
    base_position *= length

    # Create a circular base for the cone
    num_points = 30
    angles = np.linspace(0, 2 * np.pi, num_points)
    x_base = radius * np.cos(angles)
    y_base = radius * np.sin(angles)
    z_base = np.zeros_like(x_base)

    # Create the cone surface by connecting the base to the tip
    X, Y, Z = [], [], []
    for i in range(num_points):
        X.append([x_base[i], 0])  # Base point and tip
        Y.append([y_base[i], 0])  # Base point and tip
        Z.append([z_base[i], length])  # Base level and tip height
    
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # Rotation matrices for x, y, and z axes
    rot_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rot_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

    # Apply rotations and translate by the start point
    base_points = np.array([x_base, y_base, z_base]).T
    tip_point = np.array([0, 0, length])
    
    rotated_base = np.dot(rot_x, np.dot(rot_y, np.dot(rot_z, base_points.T))).T + np.array(start)
    rotated_tip = np.dot(rot_x, np.dot(rot_y, np.dot(rot_z, tip_point))) + np.array(start)

    # Plot the cone surface
    for i in range(num_points):
        ax.plot([rotated_base[i][0], rotated_tip[0]], [rotated_base[i][1], rotated_tip[1]], [rotated_base[i][2], rotated_tip[2]], **kw)

    # Add text at the tip if provided
    if text_tip:
        tip_text_position = rotated_tip + np.array(tip_offset)
        ax.text(tip_text_position[0], tip_text_position[1], tip_text_position[2], text_tip, fontsize=16, color=kw.get('color', 'black'))

    # Add text at the base if provided
    if text_base:
        base_text_position = np.array(start) + np.array(base_offset)
        ax.text(base_text_position[0], base_text_position[1], base_text_position[2], text_base, fontsize=16, color=kw.get('color', 'black'))

'''----------------DEFINITIONS----------------'''

def naturalSort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    natsort_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=natsort_key)

def getFilesUnderFolder(path, filetype=""):
    """
    get list of files under folder.
    will only find files of filetype if specified --
    pass in string without leading "." (ex. "root", "pkl")
    """
    if filetype:
        l = len(filetype)
        return [os.path.join(path,f) for f in os.listdir(path)
                if os.path.isfile(os.path.join(path,f))
                and f[-(l+1):] == f".{filetype}"]
    else:
        return [os.path.join(path,f) for f in os.listdir(path)
                if os.path.isfile(os.path.join(path,f))]

def getFileName(path):
    '''
    Get name of file, without file extension or parent folder
    '''
    namepieces = path.split("/")[-1].split(".")
    filename = ".".join(namepieces[:-1])
    return filename

def getFolderName(path):
    """
    find name of lowest level directory of given path.
    """
    if os.path.isfile(path):
        path = os.path.dirname(path)
    return os.path.basename(os.path.normpath(path))

def xy_phi(x, y):
    """
    map x, y coordinates to phi.
    phi == 0 on +y axis, increasing CCW when viewing from above.
    """
    xsign = np.where(np.sign(x) == 0, 1, np.sign(x))
    phi = np.where((x==0)&(y==0), 0, xsign*np.arccos(y/np.sqrt(x**2+y**2)))
    return phi

def rotate_pmtarray(points, angle, origin=(0,0)):
    """
    rotate points CCW in the xy-plane by a given angle around a given origin.
    angle should be given in radians.
    """
    ox, oy = origin
    rpos = []
    for point in points:
        px, py, pz = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        rpos.append([qx, qy, pz])
    rpos = np.vstack(rpos)
    return rpos

def modify_cmap(original_cmap, maxval):
    """
    make viridis cmap where only -1 ~ 0 is grey
    """
    viridis = plt.get_cmap(original_cmap, maxval)
    newcolor = viridis(np.linspace(0, 1, maxval+1))
    grey = np.array([32/256, 32/256, 32/256, 1])
    newcolor[:1, :] = grey
    return matplotlib.colors.ListedColormap(newcolor)

class EosVisualizer():
    """Class for Visualizing Eos Events"""

    def __init__(self, filepath=None, filetype=None):
        """ Initialize for different types of inputs """
        
        # if no filepath was given, assume data will be given manually
        if filepath == None:
            self.init_pmt_positions()
            self.filetype = "manual"

        # filepath is one file, check for file type
        elif filepath.split(".")[-1] == "h5":
            self.init_hdf5(filepath)

        elif filepath.split(".")[-1] == "root":
            self.init_root(filepath)

        # filepath is directory,
        # need user to manually specify file type to parse for.
        elif os.path.isdir(filepath):
            if filetype == None:
                raise Exception("Please specify filetype (hdf5 / root).")
            elif filetype == "hdf5":
                self.init_hdf5_multiple(filepath)
            elif filetype == "root":
                self.init_root_multiple(filepath)

        # settings
        self.threshold = -5.0 # threshold to pass to count as hit, in mV
        self.termination_ohms = 50
        self.boards = PMT_POSITIONS.boards
        self.ped_window = 30 # samples

        # initiate default for labels and markers
        self.show_lcn = False
        self.show_colorbar = True
        self.mark_dichroicons = True
        self.disabled_channels = []
        self.markers = []

        self.lcn_fontsize = 12

        dic_lcn = np.arange(7*16, 7*16+12)
        if "mc" in self.filetype:
            self.dichroicons = [self.pmtid2lcn.index(i) for i in dic_lcn]
        else:
            self.dichroicons = dic_lcn
    
    def init_pmt_positions(self):
        """ Initialize PMT positions from PMT_POSITIONS.py """

        pmtx = PMT_POSITIONS.x
        pmty = PMT_POSITIONS.y
        pmtz = PMT_POSITIONS.z
        self.init_pmt_hits(pmtx, pmty, pmtz)
    
    def init_pmt_hits(self, pmtx, pmty, pmtz):
        """ Initiate dict to store pmt hits """

        # global rotation to let top PMT array lie horizontally on EventDisplay
        self.pmtpos = rotate_pmtarray(np.stack((pmtx, pmty, pmtz), axis=1), np.pi/2.)

        # initiate dictionary to store charges
        self.pmt_hits = {}
        for lcn in range(len(self.pmtpos)):
            if self.pmtpos[lcn][2] == -1: # not a PMT
                continue
            self.pmt_hits[lcn] = 0
        
        self.LCNs = list(self.pmt_hits.keys())

    def init_hdf5(self, filepath):
        self.init_pmt_positions()
        self.file = h5py.File(filepath, "r")
        b0 = list(self.file.keys())[0]
        ch0 = [key for key in self.file[b0].keys() if "ch" in key][0]
        self.nEvents = len(self.file[b0][ch0]['samples'])
        self.filetype = "hdf5"
    
    def init_hdf5_multiple(self, filepath):
        self.init_pmt_positions()
        self.file = naturalSort(getFilesUnderFolder(filepath, "h5"))
        self.nFiles = len(self.file)
        self.filetype = "hdf5_multiple"

    def init_root(self, filepath):
        self.file = ROOT.TFile.Open(filepath)
        meta = self.file.Get("meta")
        meta.GetEntry(0)

        try:
            self.tree = self.file.Get("output")
            self.filetype = "mc"
            pmtx = list(meta.pmtX)
            pmty = list(meta.pmtY)
            pmtz = [z-170 for z in list(meta.pmtZ)]
            self.pmtid2lcn = list(meta.pmtChannel)
        except:
            self.tree = self.file.Get("events")
            self.filetype = "data"
            pmtx = list(meta.pmtx)
            pmty = list(meta.pmty)
            pmtz = list(meta.pmtz)

        self.nEvents = self.tree.GetEntries()
        self.init_pmt_hits(pmtx, pmty, pmtz)

    def init_root_multiple(self, filepath):
        self.file = naturalSort(getFilesUnderFolder(filepath, "root"))
        self.nFiles = len(self.file)
        f0 = ROOT.TFile.Open(self.file[0])
        meta = f0.Get("meta")
        meta.GetEntry(0)

        try:
            tree = f0.Get("output")
            self.filetype = "mc_multiple"
            pmtx = list(meta.pmtX)
            pmty = list(meta.pmtY)
            pmtz = [z-170 for z in list(meta.pmtZ)]
            self.pmtid2lcn = list(meta.pmtChannel)
        except:
            tree = f0.Get("events")
            self.filetype = "data_multiple"
            pmtx = list(meta.pmtx)
            pmty = list(meta.pmty)
            pmtz = list(meta.pmtz)
            
        self.init_pmt_hits(pmtx, pmty, pmtz)

    def load_hits_from_file(self, events, useCharge=False):
        """
        load hit/charge information from files for given list of events.
        any event # that exceeds total number of events will be ignored.
        """
        if self.filetype == "hdf5":
            self.load_from_hdf5(self.file, events)
        elif self.filetype == "mc":
            self.load_from_mc(self.tree, events, useCharge=useCharge)
        elif self.filetype == "data":
            self.load_from_data(self.tree, events, useCharge=useCharge)
        elif "multiple" in self.filetype:
            events_left, offset = events, 0
            for filepath in self.file:
                if len(events_left) == 0:
                    break
                if self.filetype == "hdf5_multiple":
                    with h5py.File(filepath, "r") as f:
                        nEvents = self.load_from_hdf5(f, events_left)
                if self.filetype == "mc_multiple":
                    f = ROOT.TFile.Open(filepath)
                    tree = f.Get("output")
                    nEvents = self.load_from_mc(tree, events_left, useCharge=useCharge)
                    f.Close()
                if self.filetype == "data_multiple":
                    f = ROOT.TFile.Open(filepath)
                    tree = f.Get("events")
                    nEvents = self.load_from_data(f, events_left, useCharge=useCharge)
                    f.Close()
                events_left = [ev-nEvents for ev in events_left if ev>=nEvents]
                offset += nEvents

    def load_from_hdf5(self, f, events):
        for b in list(f.keys()):
            b_id = self.boards[b]
            bits = f[b].attrs['bits']
            channels = [ch for ch in list(f[b].keys()) if ch[:2] == "ch"] 

            for ch in channels:
                ch_num = int(ch.replace("ch",""))
                lcn = ch_num + b_id*16
                if not lcn in self.LCNs:
                    continue

                dynamic_range = f[b][ch].attrs['dynamic_range']
                dv = dynamic_range/np.power(2, bits)
                data = np.array(f[b][ch]["samples"], dtype=np.float32)
                nEvents = len(data)

                for ev in events:
                    if ev >= nEvents:
                        break
                    pedestal = np.mean(data[ev][0:self.ped_window])
                    voltage = ((data[ev])-pedestal) * dv
                    if any(voltage < self.threshold):
                        self.pmt_hits[lcn] += 1
        return nEvents
    
    def load_from_mc(self, tree, events, useCharge=False):
        nEvents = tree.GetEntries()
        for ev in events:
            if ev >= nEvents:
                break
            tree.GetEntry(ev)
            hitPMTs = list(tree.hitPMTID)
            if useCharge:
                charges = list(tree.hitPMTCharge)
        
            for (i, pmtid) in enumerate(hitPMTs):
                if useCharge:
                    self.pmt_hits[pmtid] += charges[i]
                else:
                    self.pmt_hits[pmtid] += 1
        return nEvents
    
    def load_from_data(self, tree, events, useCharge=False):
        nEvents = tree.GetEntries()
        for ev in events:
            if ev >= nEvents:
                break
            tree.GetEntry(ev)
            hitLCNs = list(tree.lcn)
            if useCharge:
                charges = list(tree.charge)
        
            for (i, lcn) in enumerate(hitLCNs):
                if useCharge:
                    self.pmt_hits[lcn] += charges[i]
                else:
                    self.pmt_hits[lcn] += 1
        return nEvents

    def sorthits(self):
        """
        sort hits in self.pmt_hits into dictionary
        based on position in Eos (top, side, bottom lo and hi)
        """
        hits    = {"top":[], "side":[], "dic_hi":[], "dic_lo":[]}
        charges = {"top":[], "side":[], "dic_hi":[], "dic_lo":[]}

        # PMT Arrays are delineated by PMT z-position
        for lcn, q in self.pmt_hits.items():
            pos = self.pmtpos[lcn]
            if lcn in self.disabled_channels:
                q = -1 # negative value will be marked as grey
            if pos[2] > 900:
                hits["top"].append(pos)
                charges["top"].append(q)
            elif pos[2] > -700 and pos[2] < 900:
                hits["side"].append(pos)
                charges["side"].append(q)
            elif pos[2] < -700 and pos[2] > -1300:
                hits["dic_hi"].append(pos)
                charges["dic_hi"].append(q)
            elif pos[2] < -1300:
                hits["dic_lo"].append(pos)
                charges["dic_lo"].append(q)

        for loc in hits.keys():
            hits[loc] = np.array(hits[loc])
            charges[loc] = np.array(charges[loc])

        return hits, charges

    def clear_hits(self):
        """clear pmt hit charge dict after plotting"""
        for hit, q in self.pmt_hits.items():
            self.pmt_hits[hit] = 0
    
    def load_hits(self, pmt_hits):
        """
        manually feed in hits.
        pmt_hits should be a dict with LCNs as keys and nhits as values.
        """
        for lcn, q in self.pmt_hits.items():
            try:
                self.pmt_hits[lcn] = q
            except:
                pass
    
    def disable_channels(self, channels):
        """ pass in list of channels to disable """
        for ch in channels:
            if not ch in self.disabled_channels:
                self.disabled_channels.append(ch)
    
    def add_marker(self, x, y, z, marker="*", color="magenta", size=2000):
        """ add marker to event display """
        m = {'pos':(x,y,z), 'm':marker, 'c':color, 's':size}
        self.markers.append(m)

    def plot_events(self, nEvents=-1, figpath=None, useCharge=False):
        """
        Display multiple Eos events overlayed
        """

        if not self.filetype == "manual":
            if nEvents < 0:
                if "multiple" in self.filetype:
                    maxEv = 1000000
                else:
                    maxEv = self.nEvents
            else:
                maxEv = nEvents
            self.load_hits_from_file(np.arange(0,maxEv,1), useCharge=useCharge)

        hits, charges = self.sorthits()
        self.plot(hits, charges, figpath)
        self.clear_hits()
    
    def plot_single_event(self, event=-1, figpath=None, useCharge=False):
        """
        Display single Eos event
        """
        if event < 0:
            try:
                events = [np.random.randint(self.nEvents)]
            except:
                events = [np.random.randint(self.nFiles)]
        else:
            events = [event]
        self.load_hits_from_file(events, useCharge=useCharge)

        hits, charges = self.sorthits()
        self.plot(hits, charges, figpath)
        self.clear_hits()

    def plot(self, hits, charges, figpath=None):
        # Calculate radius for side PMT display aspect
        r = np.average(np.sqrt(hits["side"][:,0]**2+hits["side"][:,1]**2))

        # pmt marker size
        pmtsize = 200

       # Find highest charge pmt for colorbar
        all_charges = np.concatenate([charges["top"], charges["side"], charges["dic_hi"], charges["dic_lo"]])
        maxval = max(all_charges)
        if self.disabled_channels:
            norm = matplotlib.colors.Normalize(vmin=-1, vmax=maxval)
            cmap = modify_cmap('viridis', maxval)
        else:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=maxval)
            cmap = plt.get_cmap('viridis')

        # Display Canvas and axes
        fig = plt.figure(figsize=(18, 14), facecolor='black')
        gs = fig.add_gridspec(3, 2) # GridSpec for original layout
        
        # Create 3D axes
        ax = fig.add_subplot(111, projection='3d')
        
        # Set view angle:
        ax.view_init(vertical_axis = "y")  
        #vertical_axis = "y"
         #At the moment the side hits and top/bottom hits can't be plotted on the same graph and make a coherent shape. That's something else I'm working on fixing. If you comment out paragraphs under the "plot x hit" comment, they'll work individually
        
        # Plot side hits
        if len(hits["side"]) > 0:
               # Calculate radius and angle
               phi = xy_phi(hits["side"][:,0], hits["side"][:,1])
               radius = np.average(np.sqrt(hits["side"][:,0]**2 + hits["side"][:,1]**2))
    
               # Convert polar coordinates to Cartesian coordinates for a curved surface
               r = 4 * radius
               x_pos = r * np.cos(phi)
               y_pos = r * np.sin(phi)
    
               # Plot the side PMTs
               ax.scatter(x_pos, hits["side"][:,2], y_pos, 
                      s=pmtsize, c=charges["side"], norm=norm, cmap=cmap, label='Side Hits')
               #EosViz.add_marker(x=-502.0, y=870.4, z=571.5, marker="*", color="red", size=200)
        
        '''# Plot top hits
        if len(hits["top"]) > 0:
            ax.scatter(hits["top"][:,0], hits["top"][:,1], hits["top"][:,2],
                   s=pmtsize, c=charges["top"], norm=norm, cmap=cmap, label='Top Hits')    
                   
        # Plot dic_hi hits
        if len(hits["dic_hi"]) > 0:
            ax.scatter(hits["dic_hi"][:,0], hits["dic_hi"][:,1], hits["dic_hi"][:,2], 
                   s=pmtsize, c=charges["dic_hi"], norm=norm, cmap=cmap, label='DIC HI Hits')

        #np.ones(len(hits["dic_hi"]))*1000, np.ones(len(hits["dic_lo"]))*-1000,(goes between [:,1] and s=pmtsize when [:,2] is removed
        #this is a note to myself I can explain it if it doesn't make sense
        
        # Plot dic_lo hits
        if len(hits["dic_lo"]) > 0:
            ax.scatter(hits["dic_lo"][:,0], hits["dic_lo"][:,1], hits["dic_lo"][:,2], 
                   s=pmtsize, c=charges["dic_lo"], norm=norm, cmap=cmap, label='DIC LO Hits')'''

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Insert colorbar
        if self.show_colorbar:
            cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.8])
            cb = fig.colorbar(ax.collections[0], cax=cbar_ax)
            cbar_ax.set_ylim(bottom=0)
            fg_color = "white"
            cb.ax.yaxis.set_tick_params(color=fg_color, labelsize=24)
            cb.outline.set_edgecolor(fg_color)
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)

        '''# Mark dichroicons
        if self.mark_dichroicons:
            dic = np.array([self.pmtpos[lcn] for lcn in self.dichroicons])
            ax.scatter(dic[:,0], dic[:,1], dic[:,2], s=pmtsize*2, c="black",
                   edgecolor="goldenrod", linewidth=3, zorder=-1, label='Dichroicons')'''

        # Label LCNs
        if self.show_lcn:
            labeloffset = 60
            for pmtid in self.LCNs:
                pmtpos = self.pmtpos[pmtid]
                ax.text(pmtpos[0], pmtpos[1], pmtpos[2], str(pmtid), color="w", size=self.lcn_fontsize, ha="center")

        # Draw markers
        for m in self.markers:
            mpos = rotate_pmtarray([m['pos']], np.pi/2.)[0]
            ax.scatter(mpos[0], mpos[1], mpos[2], marker=m['m'], c=m['c'], s=m['s'], label='Markers')

        # Add legend
        # ax.legend()

        # Save and show
        if figpath:
            plt.savefig(figpath, dpi=300)
            print(f"Plot saved to {figpath}")
        else:
            plt.show()
            
        # Overlay a 3D arrow without blocking the data
        
        arrow3d(
            ax, # ax_3d: the 3D axis object where the arrow will be plotted.
            start=[-502.0, 870.4, 571.5], # start: the starting point coordinates (x, y, z) of the arrow.
            direction_ar=[0, 1, 0], # direction: vector indicating the direction of the arrow.
            length=500, # length: length of the arrow from base to tip.
            width=50, # width: width of the arrow's shaft.
            head=0.4, # head: proportion of the total length that the head occupies.
            headwidth=2, # headwidth: multiplier that determines the width of the arrow's head relative to the shaft.
            theta_x=38, # theta_x: rotation angle around the x-axis in degrees. 90 = in -y directuion. -90 = in y direction.
            theta_y=0, # theta_y: rotation angle around the y-axis in degrees. -90 = in -x direction. 90 = in positive x direction
            theta_z=0, # theta_z: rotation angle around the z-axis in degrees.
            #(0, 0, 0) is along +z axis. 
            color='green', # color: color of the arrow.
            text_tip="", # text_tip: text label placed near the tip of the arrow.
            text_base="", # text_base: text label placed near the base of the arrow.
            tip_offset=(-7500, 3000, 0), # tip_offset: moves the tip text relative to the tip's position.
            base_offset=(1000, 0, 0), # base_offset: moves the base text relative to the base's position.
            alpha=0.5 # alpha: transparency of the arrow, where 1 is opaque and 0 is fully transparent.
            )

        # Save and show the plot
        if figpath:
           plt.savefig(figpath, dpi=300)
           print(f"Plot saved to {figpath}")
        else:
           plt.show()
        
        cone3d(
            ax,  # ax: the 3D axis object where the cone will be plotted.
            start=[-502.0, 870.4, 571.5],  # start: the starting point coordinates (x, y, z) of the cone.
            direction_cn=[0, 1, 0],  # direction: vector indicating the direction of the cone.
            length=500,  # length: length of the cone from base to tip.
            radius=50,  # radius: base radius of the cone.
            theta_x=-38,  # theta_x: rotation angle around the x-axis in degrees.
            theta_y=0,  # theta_y: rotation angle around the y-axis in degrees.
            theta_z=0,  # theta_z: rotation angle around the z-axis in degrees.
            color='green',  # color: color of the cone.
            text_tip="",  # text_tip: text label placed near the tip of the cone.
            text_base="",  # text_base: text label placed near the base of the cone.
            tip_offset=(-7500, 3000, 0),  # tip_offset: moves the tip text relative to the tip's position.
            base_offset=(1000, 0, 0),  # base_offset: moves the base text relative to the base's position.
            alpha=0.5  # alpha: transparency of the cone, where 1 is opaque and 0 is fully transparent.
            )
    
        # Save and show the plot
        if figpath:
           plt.savefig(figpath, dpi=300)
           print(f"Plot saved to {figpath}")
        else:
           plt.show()
           
'''----------------IF STATEMENTS + RUN PLOT (?)----------------'''       

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("f", action="store", help="pass in ntuple file to plot")
    parser.add_argument("-s", "--single", action="store_true", help="plot event display of single event. default is to plot multiple")
    parser.add_argument("-c", "--useCharge", type=bool, default=True, help="use charge? uses nhit if set to False")
    parser.add_argument("-e", "--events", type=int, default=-1, help="number of events to be overlayed, when plotting multiple events")
    args = parser.parse_args()
    
    plotdir = "./EventDisplays/"
    try: os.makedirs(plotdir)
    except: pass

    EosViz = EosVisualizer(args.f)
    filename = getFileName(args.f)
    figpath = os.path.join(plotdir, filename+".png")

    if args.single:
        EosViz.plot_single_event(useCharge=args.useCharge, figpath=figpath)
    else:
        EosViz.plot_events(nEvents=args.events, useCharge=args.useCharge, figpath=figpath)
        
EosVis = EosVisualizer("/path/to/file")
EosVis.plot_single_event(event=0, figpath="")
