import os
import sys
import argparse

def parse_args():
    
    # Create the parser
    parser = argparse.ArgumentParser(
        description='|---- Arguments ----|'
    )
    
	# Class initializer args
    parser.add_argument(
        'data', type=str, 
        help='Input data must be in .csv format. Also rows should correspond '+ 
             'to cells and columns to genes. If the reveerse is true, use '+ 
             'the transpose (-t --transpose) option'
	)
    parser.add_argument(
        'labels', type=str,
        help='Cluster labels must be in .csv format and match the number' +
        'of cells.'
   	)   
    parser.add_argument(
        '-genann', dest='gene_ann', metavar='',
    	action='store', help='Gene Annotation file (default: None)'
	)
    parser.add_argument(
        '-v', '--verbose', dest='verbose',
        action='store_true', help='Transpose input matrix (default: False)'
    )
    
    # Read data args
    parser.add_argument(
		'-sep', '--readseperator', type=str,
		dest='sep', action='store', default=',', metavar='', 
		help='Delimiter to use.  (default: , )'
	)
    parser.add_argument(
        '-head', '--header', type=int,
        dest='header', action='store', default=0, metavar='',
        help='If header are on line 1, set it to 0. Set to None if headers '+
        'are not available. (default: %(default)s))'
    )
    parser.add_argument(
		'-incol', '--indexcol', type=int,
		dest='in_col', action='store', default=0, metavar='', 
		help='Set to 0 if the row indexes are in the 1st column. If the index '+
		'does not exist, set it to None. (default: %(default)s))'
	)
    
    #--- Gene filtering args
    parser.add_argument(
        '-thr', '--expthres', type=float,
        dest='in_cls_thres',  action='store', metavar='', 
        help='Intra-cluster expression threshold to be used in gene filtering'+
        ' (default: None)'
    )
    parser.add_argument(
        '-im', '--imexp', type=int,
        dest='im_exp', default=10, metavar='', 
        action='store', help='Significance of out-of-cluster expression rate '+
        '(default: %(default)s))'
    )
    parser.add_argument(
        '-nsel', '--nofsel', type=int,
        dest='nof_sel', default=10, metavar='', 
        action='store', help='Number of genes remaining for each cluster after'+
        ' filtering (default: %(default)s))'
    )
    parser.add_argument(
        '-nolnorm', '--nolognorm', 
        dest='log_norm', action='store_false', 
        help='Log normalization status in gene filtering (default: True)'
    )
    
    #- Select marker args
    parser.add_argument(
        '-nmark', '--nofmarkers', type=int,
        dest='nof_markers', action='store', default=5, metavar='', 
        help='Number of markers to be selected for each cluster'+
        ' (default: %(default)s))'
    )
    parser.add_argument(
        '-ncore', '--nofcores', type=int,
        dest='n_cores', action='store', default=-2, metavar='', 
        help='Number of cores to use (default: %(default)s))' 
    )
    parser.add_argument(
        '-dyn', '--dynprog', 
        dest='dyn_prog', action='store_true', 
        help='Dynamic programming option for gene selection (default: False)'
    )
        
    #-- Plotting args
    parser.add_argument(
        '-dot', '--dotplot', dest='dotplot', action='store_true',
        help='Dot plotting status (default: False)'
    )
    parser.add_argument(
        '-tsne', '--marktsne', dest='markertsne', action='store_true',
        help='T-SNE plotting status (default: False)'
    )
    parser.add_argument(
        '-heat', '--markheat', dest='markerheat', action='store_true',
        help='Heatmap plotting status (default: False)'
    )
    parser.add_argument(
        '-knn', '--knnconf', dest='knnmark', action='store_true',
        help='K-NN classiification and confusion matrix plotting status'+
        ' (default: False)'
    )
    parser.add_argument(
        '-t', '--transpose', dest='transpose', action='store_true', 
        help='If the data matrix is gene*cell, use (default: False)'
	)
    parser.add_argument(
        '-wrt', '--writeres', dest='write_res', action='store_true', 
        help='If you want to print the results to the directory where the'+
        ' data is, use (default: False)'
	)
        
    args = parser.parse_args()
    return vars(args)
       
def main():
    args = parse_args()
    from . import _cli
    _cli.run_cl_args(args)


