#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:40:21 2025

@author: kiurujul
"""
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
import scipy as scp
import argparse as arg
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
Example execution:
    ./mc_strand_builder.py -d representative_structures -t representative_structures/cluster1_representative.pdb \
    -p cluster_probabilities.dat -f cluster{1..20}_representative.pdb --interfaces 1 13 --seed 123456 --max 100 --tol 0 --plane 0 --plot -o mc_strand.pdb
"""

#TODO
#add option to constrain monomer centers of mass into the same (any) plane

# plotting settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
#matplotlib.rc("mathtext", fontset = "cm", rm = "cm")
matplotlib.rcParams['text.usetex'] = True
w = 2.5
l = 10
fs = 28
r = 10
color = 'k'

# TODO consider making these command line arguments
frac = 0.80
percentile = 0
lim_cum_prob = 0.5
max_tries = 1000
tol_increment = 20

def read_interfaces(dir, files, top, version):
    """
    Read structure of interfaces.
    
    Parameters
    ----------
    dir : str
        Directory with input data files.
    files : list of str
        Input data files.
    top : str
        Topology of input molecule.

    Returns
    -------
    interfaces : list of mda.Universe
        List containing an mda.Universe of each of the interfaces.

    """
    
    if version == 'pdb':
        interfaces = []
    
        for f in files:
            interfaces.append(mda.Universe(top, dir + f))
    
        #for i, f in enumerate(files):
        #    if i in [1,5,8,10]:
        #        interfaces.append(mda.Universe(top, dir + f))
    elif version == 'xtc':
        interfaces = mda.Universe(top, dir + files)
    else:
        raise ValueError
    
    return interfaces
    
def read_probabilities(file, interfaces):
    """
    Read probabilities of interfaces.

    Parameters
    ----------
    file : str
        File containing probablities of interfaces.
    interfaces : list of int
        List containing the interfaces to be used.

    Returns
    -------
    probabilities : np.array
        Numpy array containing probablities of interfaces.

    """
    # TODO standardize the probability files and remove the error catch
    try:
        probabilities = np.loadtxt(file, delimiter = ';', usecols = 1)
    except:
        probabilities = np.loadtxt(file, delimiter = ';', usecols = 1, skiprows = 1)
    
    # normalize probability
    probabilities = probabilities / np.sum(probabilities)
    # sort indices from largest probability to smallest
    ind = np.argsort(probabilities)[::-1]
    # TODO removed for now, add back later with an if statement and command line argument telling
    # which method to use to choose the accepted interfaces
    '''
    # calculate cumulative probability
    cum_prob = np.cumsum(probabilities[ind])
    # take the indices whose cumulative sum starting from the largest probability is less than 0.5 plus the interface
    # for which the cumulative probability goes over 0.5. sort this indices from smallest index to largest
    accepted_interfaces = np.sort(np.concatenate((ind[cum_prob < lim_cum_prob], ind[cum_prob >= lim_cum_prob][0:1])))
    '''
    # get accepted interfaces
    if interfaces[0] == -1:
        accepted_interfaces = np.arange(probabilities.size)
    else:
        # make indices out of interface numbers
        accepted_interfaces = np.array(interfaces) - 1
    # create mask which hides all non-accepted interfaces
    mask = np.zeros(len(probabilities), dtype = bool)
    mask[accepted_interfaces] = True
    print(accepted_interfaces)
    
    probabilities = probabilities[mask]
    
    #return probabilities[[1,5,8,10]]
    return probabilities / np.sum(probabilities), accepted_interfaces

def proj(v,normal_vector):
    return v - np.dot(v, normal_vector) * normal_vector / np.linalg.norm(normal_vector)**2

def calculate_angle(chainA, chainB):
    helix1A = chainA.select_atoms('resid 2 to 26')
    helix1B = chainB.select_atoms('resid 2 to 26')

    # calculate centers of mass of both chains and the first alpha helix
    comA = chainA.center_of_mass()
    comB = chainB.center_of_mass()
    comH1A = helix1A.center_of_mass()
    comH1B = helix1B.center_of_mass()
    # principal axes
    paA = chainA.principal_axes()
    paB = chainB.principal_axes()

    paxisA = 0
    if np.abs(paA[1,2]) >= np.abs(paA[0,2]):
        if np.abs(paA[1,2]) > np.abs(paA[2,2]):
            paxisA = 1
        else:
            paxisA = 2
    else:
        if np.abs(paA[2,2]) >= np.abs(paA[0,2]):
            paxisA = 2

    paA = paA * np.sign(paA[paxisA,2])

    ref_vecA = np.zeros(3)
    ref_vecA[0] = np.sqrt( 1 / ( 1 + (paA[paxisA,0]/paA[paxisA,2])**2 ) )
    ref_vecA[2] = - ref_vecA[0] * paA[paxisA,0] / paA[paxisA,2]

    paxisB = 0
    if np.abs(paA[1,2]) >= np.abs(paA[0,2]):
        if np.abs(paA[1,2]) > np.abs(paA[2,2]):
            paxisB = 1
        else:
            paxisB = 2
    else:
        if np.abs(paA[2,2]) >= np.abs(paA[0,2]):
            paxisB = 2

    paB = paB * np.sign(paB[paxisB,2])

    ref_vecB = np.zeros(3)
    ref_vecB[0] = np.sqrt( 1 / ( 1 + (paA[paxisB,0]/paA[paxisB,2])**2 ) )
    ref_vecB[2] = - ref_vecB[0] * paB[paxisB,0] / paB[paxisB,2]

    # calculate vectors combining the centers of mass and project them to the surface perpendicular to paxis
    comAtoB = proj(comB - comA, paA[paxisA,:])
    comAtoH1A = proj(comH1A - comA, paA[paxisA,:])
    comBtoA = proj(comA - comB, paB[paxisB,:])
    comBtoH1B = proj(comH1B - comB, paB[paxisB,:])

    # get the angles between the vectors
    lenAB = np.linalg.norm(comAtoB)
    lenAH1A = np.linalg.norm(comAtoH1A)
    len_refA = np.linalg.norm(ref_vecA)
    angle1 = -180/np.pi * np.arccos( np.dot(ref_vecA, comAtoH1A) / (len_refA * lenAH1A) )
    angle2 = -180/np.pi * np.arccos( np.dot(ref_vecA, comAtoB) / (len_refA * lenAB) )

    # make the angle go from -180 to 180 instead of 0 to 180
    if np.dot( np.cross(ref_vecA, comAtoH1A), paA[paxisA,:] ) < 0:
        angle1 = -angle1

    # make the angle go from -180 to 180 instead of 0 to 180
    if np.dot( np.cross(ref_vecA, comAtoB), paA[paxisA,:] ) < 0:
        angle2 = -angle2

    angleA = ( angle1 - angle2 + 47.5 ) % 360

    # get the angles between the vectors
    lenBA = np.linalg.norm(comBtoA)
    lenBH1B = np.linalg.norm(comBtoH1B)
    len_refB = np.linalg.norm(ref_vecB)
    angle1 = -180/np.pi * np.arccos( np.dot(ref_vecB, comBtoH1B) / (len_refB * lenBH1B) )
    angle2 = -180/np.pi * np.arccos( np.dot(ref_vecB, comBtoA) / (len_refB * lenBA) )

    # make the angle go from -180 to 180 instead of 0 to 180
    if np.dot( np.cross(ref_vecB, comBtoH1B), paB[paxisB,:] ) < 0:
        angle1 = -angle1

    # make the angle go from -180 to 180 instead of 0 to 180
    if np.dot( np.cross(ref_vecB, comBtoA), paB[paxisB,:] ) < 0:
        angle2 = -angle2

    angleB = ( angle1 - angle2 + 47.5 ) % 360
    
    return angleA, angleB


def create_strand(top, monomers):
    """
    Create strand consisting of one monomer.

    Parameters
    ----------
    top : str
        Topology of input molecule.
    monomers : list of AtomGroup
        List containing all added monomers separately.

    Returns
    -------
    strand : AtomGroup
        AtomGroup containing one monomer.

    """
    # vector to align the principal axis of the new monomer to
    # change to be calculated based on the given plane
    align_vec = np.array([0,0,1])
    
    mol = mda.Universe(top)
    seg = mol.add_Segment(segid = '0')
    strand = mol.select_atoms('chainID A')
    
    # calculate the principal axes of the monomer
    p = strand.principal_axes()
    # choose the principal axis most aligned with the z-axis
    paxis = 0
    if np.abs(p[1,2]) >= np.abs(p[0,2]):
        if np.abs(p[1,2]) > np.abs(p[2,2]):
            paxis = 1
        else:
            paxis = 2
    else:
        if np.abs(p[2,2]) >= np.abs(p[0,2]):
            paxis = 2
    # make sure the rotation will not flip the monomer around
    align_vec[2] = np.sign(p[paxis,2])
    # calculate the rotation matrix that will align the principal axis of the new monomer with z-hat
    v = np.cross(p[paxis,:], align_vec)
    sin = np.linalg.norm(v)
    cos = np.dot(p[paxis,:], align_vec)
    v_rot = np.zeros((3,3))
    v_rot[0,1] = -v[2]
    v_rot[0,2] = v[1]
    v_rot[1,0] = v[2]
    v_rot[1,2] = -v[0]
    v_rot[2,0] = -v[1]
    v_rot[2,1] = v[0]
    R = np.eye(3) + v_rot + v_rot @ v_rot * (1 - cos) / sin**2
    # rotate the new monomer so that the principal axis is aligned with z-hat
    strand.rotate(R, point = strand.center_of_mass())
    
    # calculate the center of mass of the first monomer
    com = strand.center_of_mass()
    # translate the first monomer so that its center of mass is on the wanted plane with x=0 and y=0
    strand.translate(-com)
    strand.residues.segments = seg
    strand = mol.select_atoms('segid 0')
    monomers.append(strand)
    
    return strand


def test_new_interface(strand, interface, chain, monomer, N_monomers, tol, monomers, cutoff=4.0):
    """
    Test if the new interface can be added to the strand.

    Parameters
    ----------
    strand : AtomGroup
        AtomGroup containing the strand before.
    interface : mda.Universe
        mda.Universe containing the interface that is tried to be added.
    chain : int
        Integer representing the chain of the interface to fit the monomer on: 0 for chain A and 1 for chain B.
    monomer : int
        Integer representing the monomer to fit the interface on.
    N_monomers : int
        Number of monomers in the strand before.
    tol : int
        Maximum number of clashes to tolerate from new interface.
    monomers : list of AtomGroup
        List containing all added monomers separately.
    cutoff : float, optional
        Distance cutoff for clash calculation. The default is 4.0.

    Returns
    -------
    clashes : int
        Number of clashes.
    AtomGroup
        AtomGroup containing the strand after.
    N_monomers : int
        Number of monomers in the strand after.

    """
    # vector to align the principal axis of the new monomer to
    # change to be calculated based on the given plane
    align_vec = np.array([0,0,1])
    
    for i in range(2):
        c = 'A' if chain == 0 else 'B'
        temp_interface = interface.copy()
        
        # align new interface to the selected monomer using the selected chain
        align.alignto(temp_interface, strand, select = {'mobile': f'chainID {c} and backbone', 'reference': f'segid {monomer} and backbone'})
        
        # get the atoms belonging to the monomer that is being tried to be added
        new_monomer = temp_interface.select_atoms(f'not chainID {c}')
        
        # get poisitions of each CA atom of the non aligned selections
        points_strand = strand.select_atoms(f'not segid {monomer} and name CA').positions
        points_new = new_monomer.select_atoms('name CA').positions
        
        dist_mat = scp.spatial.distance.cdist(points_strand, points_new, 'euclidean')
        clashes = np.count_nonzero(dist_mat <= cutoff)
        
        if clashes <= tol:
            # calculate the principal axes of the monomer
            p = new_monomer.principal_axes()
            # choose the principal axis most aligned with the z-axis
            paxis = 0
            if np.abs(p[1,2]) >= np.abs(p[0,2]):
                if np.abs(p[1,2]) > np.abs(p[2,2]):
                    paxis = 1
                else:
                    paxis = 2
            else:
                if np.abs(p[2,2]) >= np.abs(p[0,2]):
                    paxis = 2
            # make sure the rotation will not flip the monomer around
            align_vec[2] = np.sign(p[paxis,2])
            # calculate the rotation matrix that will align the principal axis of the new monomer with z-hat
            v = np.cross(p[paxis,:], align_vec)
            sin = np.linalg.norm(v)
            cos = np.dot(p[paxis,:], align_vec)
            v_rot = np.zeros((3,3))
            v_rot[0,1] = -v[2]
            v_rot[0,2] = v[1]
            v_rot[1,0] = v[2]
            v_rot[1,2] = -v[0]
            v_rot[2,0] = -v[1]
            v_rot[2,1] = v[0]
            R = np.eye(3) + v_rot + v_rot @ v_rot * (1 - cos) / sin**2
            # rotate the new monomer so that the principal axis is aligned with z-hat
            new_monomer.rotate(R, point = new_monomer.center_of_mass())
            
            # calculate the center of mass of the new monomer
            com = new_monomer.center_of_mass()
            # translate the new monomer so that its center of mass is on the wanted plane
            new_monomer.translate([0,0,-com[2]])
            
            seg = temp_interface.add_Segment(segid = f'{N_monomers}')
            new_monomer.residues.segments = seg
            new_monomer = temp_interface.select_atoms(f'segid = {N_monomers}')
            N_monomers += 1
            strand = mda.core.universe.Merge(strand.select_atoms('all'), new_monomer)
            monomers.append(new_monomer)
            return clashes, strand.select_atoms('all'), N_monomers, chain, com[:2]
        else:
            chain = (chain + 1) % 2
        
    return clashes, strand.select_atoms('all'), N_monomers, chain, 0

def MC_loop_pdb(args):
    """
    MC loop for input data in pdb format.

    Parameters
    ----------
    args : argparse.parser
        Command line arguments
    interfaces : list of mda.Universe
        List constaing the mda.Universe of each interface read from pdbs.
    interface_probabilities : np.ndarray of floats
        Probability associated with each interface.
    interface_indices : np.ndarray of ints
        List of indices from 0 to N_interfaces-1.
    rng : np.random.default_rng()
        Random number generator.

    Returns
    -------
    None.

    """
    
    # read input structures (interfaces) and probabilities
    interfaces = read_interfaces(args.d, args.f, args.t, 'pdb')
    interface_probabilities, accepted_interfaces = read_probabilities(args.p, args.interfaces)
    interface_indices = np.arange(len(accepted_interfaces))
    draw_limit = np.percentile(interface_probabilities, percentile)
    print(accepted_interfaces + 1)
    print(interface_probabilities)
    
    tol = args.tol
    # increase the number of allowed clashes by tol_increment until all N_monomers have been placed successfully
    # but a maximum of 5 times
    for i in range(6):
        print(tol)
        # initialize random number generator
        rng = np.random.default_rng(args.seed) if not args.seed is None else np.random.default_rng()
        
        # add first monomer
        N_monomers = 1
        monomers = []
        strand = create_strand(args.t, monomers)
        
        # start continuous rejections counter
        n_rejections = 0
        
        # statistics
        added_interfaces = np.zeros(len(accepted_interfaces), dtype = np.int64)
        mon_interfaces = [[]]
        connections = np.zeros((len(accepted_interfaces), len(accepted_interfaces)))
        coms = np.zeros((args.max, 2))
        
        if args.plot:
        # start figure of monomers
            fig, ax = plt.subplots(figsize=(12,12))
            # plot circle for first monomer
            ax.add_patch(plt.Circle((0,0), r, color = color, fill = False))
        
        while N_monomers < args.max:
            # choose monomer
            mon = rng.integers(0, N_monomers)
            # choose interface
            interface_index = rng.choice(interface_indices, p = interface_probabilities)
            interface = interfaces[accepted_interfaces[interface_index]]
            # choose which monomer to fit on
            chain = rng.integers(0, 2)
            # try adding new interface
            clashes, strand, N_monomers, chain, com = test_new_interface(strand, interface, chain, mon, N_monomers, tol, monomers)
            
            if clashes <= tol:
                #Accept
                # add statistics
                added_interfaces[interface_index] += 1
                for inter in mon_interfaces[mon]:
                    connections[inter, interface_index] += 1
                    connections[interface_index, inter] += 1
                    #print(f'Added connection between cluster {inter+1} and {interface_index+1}.')
                mon_interfaces.append([interface_index])
                mon_interfaces[mon].append(interface_index)
                coms[N_monomers-1,:] = com
    
                if args.plot:
                    #ax.plot(com[0], com[1], 'ko')
                    # calculate the average between the centers of the two monomers involved in the new interface
                    avg = np.average(coms[[mon,N_monomers-1],:], axis = 0)
                    '''
                    # add the name of the interface at the above calculated location
                    ax.annotate(f'{accepted_interfaces[interface_index]+1}a' if chain == 0 else f'{accepted_interfaces[interface_index]+1}b', avg, 
                                horizontalalignment = 'center', verticalalignment = 'center_baseline')
                    '''
                    ax.add_patch(plt.Circle(com, r, color = color, fill = False))
                    '''
                    if interface_probabilities[interface_index] >= draw_limit:
                        # plot lines connecting the two centers of the monomers, leaving a small space for the interface number (frac)
                        ax.plot((coms[mon,0], coms[mon,0] + (avg[0]-coms[mon,0])*frac), 
                                (coms[mon,1], coms[mon,1] + (avg[1]-coms[mon,1])*frac), 'k-')
                        ax.plot((coms[N_monomers-1,0] + (avg[0]-coms[N_monomers-1,0])*frac, coms[N_monomers-1,0]), 
                                (coms[N_monomers-1,1] + (avg[1]-coms[N_monomers-1,1])*frac, coms[N_monomers-1,1]), 'k-')
                    '''
                #print(f'Added interface {accepted_interfaces[interface_index]+1} to monomomer {mon}. Number of monomers {N_monomers}.')
                n_rejections = 0
    
            else:
                #Reject
                #print(f'Interface {accepted_interfaces[interface_index]+1} had {clashes} clashes with monomomer {mon}.')
                n_rejections += 1
                
                if n_rejections > max_tries:
                    print(f'Simulation reached a dead end after adding {N_monomers} monomers.')
                    break
                
                continue
    
        if args.plot:
            # finish plotting the monomers
            ax.set_xlim(np.min(coms[:,0])-r, np.max(coms[:,0])+r)
            ax.set_ylim(np.min(coms[:,1])-r, np.max(coms[:,1])+r)
            ax.set_aspect('equal')
            fig.tight_layout()
            fig.savefig(args.o[:-4] + f'_2d_map_tol_{tol}_seed_{args.seed}.pdf')
        '''    
            # plot histogram of distribution of added interfaces
            hist, hist_ax = plt.subplots()
            bins = np.linspace(0.5, interface_indices.size + 0.5, interface_indices.size + 1)
            hist_ax.hist(bins[:-1], bins, weights = added_interfaces)
            hist_ax.set_xticks(interface_indices+1, accepted_interfaces+1)
            hist.savefig(args.o[:-4] + f'_hist_tol_{tol}_seed_{args.seed}.pdf')
            
            # plot 2d-histogram of connections between interfaces
            im, im_ax = plt.subplots()
            cplot = im_ax.imshow(connections, cmap = 'hot', interpolation = 'nearest', aspect = 'equal')
            divider = make_axes_locatable(im_ax)
            cax = divider.append_axes("right", size="6%", pad=0.5)  
            cbar = plt.colorbar(cplot, cax=cax)
            #im_ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length = l, width = w)
            im_ax.set_xticks(interface_indices, accepted_interfaces+1)
            im_ax.set_yticks(interface_indices, accepted_interfaces+1)
            im_ax.set_xlabel('angle chain B')
            im_ax.set_ylabel('angle chain A')
            #ax.set_title(f'Quarter {k+1}', fontsize = 2*fs) 
            #im_ax.xaxis.set_label_position('top')
            #cbar.ax.tick_params(labelsize = fs, length = l, width = w)
            cbar.set_label('number of connections')
            im.savefig(args.o[:-4] + f'_connections_tol_{tol}_seed_{args.seed}.pdf')
        
        dist_mat = scp.spatial.distance.cdist(coms[:N_monomers,:], coms[:N_monomers,:], 'euclidean')
        
        # starting from minus one since the distance to itself is zero and needs to be removed
        N_nearest_neighbors = -np.ones(N_monomers)
        nearest_neighbors = [[] for i in range(N_monomers)]
        angles = [[] for i in range(N_monomers)]
        for i in range(N_monomers):
            mask = dist_mat[i,:] <= 40 # checking which coms are closer together than 4 nm
            N_nearest_neighbors[i] += np.sum(mask)
            nearest_neighbors[i].append(coms[:N_monomers,:][mask])
            for j in np.arange(N_monomers)[mask]:
                if i != j:
                    angles[i].append(calculate_angle(monomers[i], monomers[j]))
                    
        with open('neighbors/' + args.o[:-4] + f'-N_nearest_neighbors_tol_{tol}_seed_{args.seed}.dat', 'w') as f:
            for i in range(1,9):
                if i > np.max(N_nearest_neighbors):
                    s = 0
                else:
                    s = np.sum(N_nearest_neighbors == i)
                f.write(f'{i} {s}\n')
        
        with open('angles/' + args.o[:-4] + f'-angles_tol_{tol}_seed_{args.seed}.dat', 'w') as f:
            for row in angles:
                for j in row:
                    f.write(f'{j[0]} {j[1]};')
                f.write('\n')
        
        with open('connections/' + args.o[:-4] + f'-connections_tol_{tol}_seed_{args.seed}.dat', 'w') as f:
            for row in connections:
                for j in row:
                    f.write(f'{int(j)} ')
                f.write('\n')
    
        if args.plot:
            plt.figure()
            plt.hist(N_nearest_neighbors, bins = np.linspace(np.min(N_nearest_neighbors)-0.5, np.max(N_nearest_neighbors)+0.5, 
                     int(np.max(N_nearest_neighbors) - np.min(N_nearest_neighbors)) + 2), rwidth = 0.9, edgecolor = 'k')
            plt.savefig(args.o[:-4] + '_neighbors_hist_tol_{tol}_seed_{args.seed}.pdf')
        
            # write the created strand to the output file
            strand.write(args.o[:-4] + f'_tol_{tol}_seed_{args.seed}.pdb')
        '''    
        if N_monomers != args.max:
            tol += tol_increment
        else:
            break
    
    #plt.show()

# TODO add all the features of the pdb function to the xtc function
def MC_loop_xtc(args):
    """
    MC loop for input data in xtc format.

    Parameters
    ----------
    args : argparse.parser
        Command line arguments
    interface_trajs : list of mda.Universe
        List constaing the mda.Universe of each trajectory read from xtcs.
    traj_indices : np.ndarray of ints
        List of indices from 0 to N_trajs-1.
    rng : np.random.default_rng()
        Random number generator.

    Returns
    -------
    None.

    """
    
    # read input structures (interfaces)
    interface_trajs = read_interfaces(args.d, args.f, args.t, 'xtc')
    traj_indices = np.arange(len(interface_trajs.trajectory))
    # create universe with just one dimer
    dimer = mda.Universe(args.t)

    # initialize random number generator
    rng = np.random.default_rng(args.seed) if not args.seed is None else np.random.default_rng()
    
    # add first monomer
    N_monomers = 1
    monomers = []
    strand = create_strand(args.t, monomers)
    
    # statistics
    #added_interfaces = np.zeros(len(interfaces), dtype = np.int64)
    #mon_interfaces = [[]]
    #connections = np.zeros((len(interfaces), len(interfaces)))
    coms = np.zeros((args.max, 2))
    
    # start figure of monomers
    fig, ax = plt.subplots()
    # plot circle for first monomer
    ax.add_patch(plt.Circle((0,0), r, color = color, fill = False))
    
    while N_monomers < args.max:
        # choose monomer
        mon = rng.integers(0, N_monomers)
        # choose interface in trajectory
        interface_frame = rng.choice(traj_indices)
        interface = dimer.copy()
        interface.atoms.positions = interface_trajs.trajectory[interface_frame].positions
        
        # choose which monomer to fit on
        chain = rng.integers(0, 2)
        # try adding new interface
        clashes, strand, N_monomers, chain, com = test_new_interface(strand, interface, chain, mon, N_monomers, args.tol, monomers)
        
        if clashes <= args.tol:
            #Accept
            # add statistics
            # added_interfaces[interface_index] += 1
            # for inter in mon_interfaces[mon]:
            #     connections[inter, interface_index] += 1
            #     connections[interface_index, inter] += 1
            #     print(f'Added connection between cluster {inter+1} and {interface_index+1}.')
            # mon_interfaces.append([interface_index])
            # mon_interfaces[mon].append(interface_index)
            coms[N_monomers-1,:] = com
            #ax.add_patch(plt.Circle(com, r, color = color, fill = False))
            ax.plot(com[0], com[1], 'ko')
            # calculate the average between the centers of the two monomers involved in the new interface
            avg = np.average(coms[[mon,N_monomers-1],:], axis = 0)
            # add the name of the interface at the above calculated location
            # ax.annotate(f'{interface_index+1}a' if chain == 0 else f'{interface_index+1}b', avg, 
            #             horizontalalignment = 'center', verticalalignment = 'center_baseline')
            #if interface_probabilities[interface_index] >= draw_limit:
            if True:
                # plot lines connecting the two centers of the monomers, leaving a small space for the interface number (frac)
                ax.plot((coms[mon,0], coms[mon,0] + (avg[0]-coms[mon,0])*frac), 
                        (coms[mon,1], coms[mon,1] + (avg[1]-coms[mon,1])*frac), 'k-')
                ax.plot((coms[N_monomers-1,0] + (avg[0]-coms[N_monomers-1,0])*frac, coms[N_monomers-1,0]), 
                        (coms[N_monomers-1,1] + (avg[1]-coms[N_monomers-1,1])*frac, coms[N_monomers-1,1]), 'k-')
            
            print(f'Added interface from frame {interface_frame} of trajectory to monomomer {mon}. Number of monomers {N_monomers}.')
            
        else:
            #Reject
            print(f'Interface from frame {interface_frame} of trajectory had {clashes} clashes with monomomer {mon}.')
            continue
        
    # finish plotting the monomers
    ax.set_xlim(np.min(coms[:,0])-r, np.max(coms[:,0])+r)
    ax.set_ylim(np.min(coms[:,1])-r, np.max(coms[:,1])+r)
    ax.set_aspect('equal')
    fig.tight_layout()
    
    # plot histogram of distribution of added interfaces
    # hist, hist_ax = plt.subplots()
    # bins = np.linspace(0.5, interface_indices.size + 0.5, interface_indices.size + 1)
    # hist_ax.hist(bins[:-1], bins, weights = added_interfaces)
    # hist_ax.set_xticks(interface_indices+1)
    
    # plot 2d-histogram of connections between interfaces
    # im, im_ax = plt.subplots()
    # cplot = im_ax.imshow(connections, cmap = 'hot', interpolation = 'nearest', aspect = 'equal')
    # divider = make_axes_locatable(im_ax)
    # cax = divider.append_axes("right", size="6%", pad=0.5)  
    # cbar = plt.colorbar(cplot, cax=cax)
    #im_ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length = l, width = w)
    # im_ax.set_xticks(interface_indices, interface_indices+1)
    # im_ax.set_yticks(interface_indices, interface_indices+1)
    # im_ax.set_xlabel('angle chain B')
    # im_ax.set_ylabel('angle chain A')
    #ax.set_title(f'Quarter {k+1}', fontsize = 2*fs) 
    #im_ax.xaxis.set_label_position('top')
    #cbar.ax.tick_params(labelsize = fs, length = l, width = w)
    # cbar.set_label('number of connections')
    
    # write the created strand to the output file
    strand.write(args.o)
    
    plt.show()
    

if __name__ == "__main__":    
    
    # read command line
    parser = arg.ArgumentParser()
    parser.add_argument('-d', default = './', help = 'Directory with input data files')
    parser.add_argument('-f', nargs = '+',  help = 'Input data files')
    parser.add_argument('-o', help = 'Output filename')
    parser.add_argument('-t', help = 'Topology of input molecule')
    parser.add_argument('-p', help = 'Probabilities of interfaces')
    parser.add_argument('--interfaces', nargs = '+', type = int, default = [-1], help = 'Interfaces to be used (space separated)')
    parser.add_argument('--seed', type = int, default = None, help = 'Seed for random number generator')
    parser.add_argument('--max', type = int, default = 10, help = 'Maximum number of monomers')
    parser.add_argument('--tol', type = int, default = 0, help = 'Maximum number of clashes to tolerate from new interface.')
    parser.add_argument('--plane', type = str, default = '0', help = 'Plane to restrict the centers of mass of the monomers to.')
    parser.add_argument('--plot', type = bool, default = True, action=arg.BooleanOptionalAction, help = 'Plot results.')
    args = parser.parse_args()

    # check if directory name ends with a /
    if args.d[-1] != '/':
        args.d = args.d + '/'


    MC_loop_pdb(args)
