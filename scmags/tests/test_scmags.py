#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple tests
"""

import numpy as np
import scmags as mg

#--- Test main functions
def test_filter():
    
    #- For default parameters
    baron_h1 = mg.datasets.baron_h1(verbose=False)
    baron_h1.filter_genes(nof_sel=10)   
    rem_genes = baron_h1.get_filter_genes(ind_return=True)
    len_rem = np.array([len(rem_genes[ky]) for ky in rem_genes.keys()])
    
    #- For different expresiion threshold
    baron_h1.filter_genes(in_cls_thres=0.5)   
    c_thres = list(baron_h1.get_filt_cluster_thresholds.values())
    
    assert len(len_rem) == len(np.unique(baron_h1.labels))
    assert all(np.array(len_rem) == 10)
    assert all(np.array(c_thres) == 0.5)
    
def test_sel_marker():
    
    #- For default parameters
    baron_h1 = mg.datasets.baron_h1(verbose=False)
    baron_h1.filter_genes(nof_sel=10)   
    baron_h1.sel_clust_marker()
    markers = baron_h1.get_markers(ind_return=True)
    
    assert markers.shape[0] == len(np.unique(baron_h1.labels))
    assert markers.shape[1] == 3
    assert (markers.isna() == False).all().all()
    assert (baron_h1.gene_ann[markers] == np.array(baron_h1.get_markers())).all()
    
    #- For dynamic programming
    baron_h1.sel_clust_marker(dyn_prog=True)
    markers = baron_h1.get_markers(ind_return=True)
    
    assert markers.shape[0] == len(np.unique(baron_h1.labels))
    assert markers.shape[1] == 3
    assert (markers.isna() == False).all().all()
    assert (baron_h1.gene_ann[markers] == np.array(baron_h1.get_markers())).all()
    
    #- For selection of different number of markers
    baron_h1.filter_genes(nof_sel=15)   
    baron_h1.sel_clust_marker(nof_markers=10)
    markers = baron_h1.get_markers(ind_return=True)
    
    assert markers.shape[0] == len(np.unique(baron_h1.labels))
    assert markers.shape[1] == 10
    assert (markers.isna() == False).all().all()
    assert (baron_h1.gene_ann[markers] == np.array(baron_h1.get_markers())).all()
    
    #- For the number of cores
    baron_h1.filter_genes(nof_sel=10)   
    baron_h1.sel_clust_marker(n_cores=1)
    markers = baron_h1.get_markers(ind_return=True)
    
    assert markers.shape[0] == len(np.unique(baron_h1.labels))
    assert markers.shape[1] == 3
    assert (markers.isna() == False).all().all()
    assert (baron_h1.gene_ann[markers] == np.array(baron_h1.get_markers())).all()
    
    
#--- Test Datasets 

def test_baron_h1():
    
    baron_h1 = mg.datasets.baron_h1(verbose=False)
    unq_lab = np.array(['acinar', 'activated_stellate', 'alpha', 'beta', 
                        'delta', 'ductal', 'endothelial', 'epsilon', 
                        'gamma', 'macrophage', 'mast', 'quiescent_stellate', 
                        'schwann', 't_cell'])
    
    assert baron_h1.data.shape == (1937, 20125)
    assert baron_h1.labels.shape == (1937,)
    assert baron_h1.gene_ann.shape == (20125,)
    assert any(np.unique(baron_h1.labels) == unq_lab)
    
    
def test_pollen():
    
    pollen = mg.datasets.pollen()
    unq_lab = np.array(['Hi_2338', 'Hi_2339', 'Hi_BJ', 'Hi_GW16', 'Hi_GW21', 
                        'Hi_HL60', 'Hi_K562', 'Hi_Kera', 'Hi_NPC', 'Hi_iPS'])
    
    assert pollen.data.shape == (301, 23730)
    assert pollen.labels.shape == (301,)
    assert pollen.gene_ann.shape == (23730,)
    assert any(np.unique(pollen.labels) == unq_lab)


def test_li():
    
    li = mg.datasets.li(verbose=False)
    unq_lab = np.array(['A549', 'GM12878_B1', 'GM12878_B2', 'H1437', 'H1_B1', 
                        'H1_B2', 'HCT116', 'IMR90', 'K562'])
    
    assert li.data.shape == (561, 55186)
    assert li.labels.shape == (561,)
    assert li.gene_ann.shape == (55186,)
    assert any(np.unique(li.labels) == unq_lab)
    
    
