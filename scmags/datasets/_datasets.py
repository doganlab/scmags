import pandas as pd
from pathlib import Path
from .._api import ScMags

HERE = str(Path(__file__).parent)

#--------------------------- Baron Human 1 GSE84133 --------------------------#
def baron_h1(verbose: bool = True) -> ScMags:

    d_pth = HERE + '/data/Baron_H1_Data.csv'
    l_pth = HERE + '/data/Baron_H1_Labels.csv'
    g_pth = HERE + '/data/Baron_H1_Gene_Ann.csv'
    # Read Data
    data = pd.read_csv(d_pth, sep=',', header = 0,  index_col = 0).to_numpy().T
    # Read Labels
    labels = pd.read_csv(l_pth, sep=',', header = 0,  index_col = 0).to_numpy().T
    # Read Gene Annotation
    gene_ann = pd.read_csv(g_pth, sep=',', header = 0,  index_col = 0).to_numpy().T
    
    labels = labels.reshape(data.shape[0])
    gene_ann = gene_ann.reshape(data.shape[1])
    obj = ScMags(data, labels, gene_ann, verbose = verbose)
    
    return obj

#------------------------------- Li GSE81861 ---------------------------------#
def li(verbose: bool = True) -> ScMags:
    
    d_pth = HERE + '/data/Li_Data.csv'
    l_pth = HERE + '/data/Li_Labels.csv'
    g_pth = HERE + '/data/Li_Gene_Ann.csv'
    # Read Data
    data = pd.read_csv(d_pth, sep=',', header = 0,  index_col = 0).to_numpy().T
    # Read Labels
    labels = pd.read_csv(l_pth, sep=',', header = 0,  index_col = 0).to_numpy().T
    # Read Gene Annotation
    gene_ann = pd.read_csv(g_pth, sep=',', header = 0,  index_col = 0).to_numpy().T
    
    labels = labels.reshape(data.shape[0])
    gene_ann = gene_ann.reshape(data.shape[1])
    obj = ScMags(data, labels, gene_ann, verbose = verbose)
    
    return obj
    
#------------------------------ Pollen SRP041736 -----------------------------#   
def pollen(verbose: bool = True) -> ScMags:

    d_pth = HERE + '/data/Pollen_Data.csv'
    l_pth = HERE + '/data/Pollen_Labels.csv'
    g_pth = HERE + '/data/Pollen_Gene_Ann.csv'
    # Read Data
    data = pd.read_csv(d_pth, sep=',', header = 0,  index_col = 0).to_numpy().T
    # Read Labels
    labels = pd.read_csv(l_pth, sep=',', header = 0,  index_col = 0).to_numpy().T
    # Read Gene Annotation
    gene_ann = pd.read_csv(g_pth, sep=',', header = 0,  index_col = 0).to_numpy().T
    
    labels = labels.reshape(data.shape[0])
    gene_ann = gene_ann.reshape(data.shape[1])
    obj = ScMags(data, labels, gene_ann, verbose = verbose)
    
    return obj
