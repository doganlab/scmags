import pandas as pd
from ._api import ScMags

def run_cl_args(args):
    
    data = pd.read_csv(
        args['data'], sep=args['sep'], 
        header = args['header'], index_col = args['in_col']
    )
    if args['transpose']:
        data = data.T
    labels = pd.read_csv(
        args['labels'], sep=args['sep'], 
        header = args['header'], index_col = args['in_col']
    )
    if (len(labels) != data.shape[0]):
        raise ValueError("Label count and cell count are not equal !")
    if args['gene_ann'] is not None:
        gene_ann = pd.read_csv(
            args['gene_ann'], sep=args['sep'], 
            header = args['header'], index_col = args['in_col']
        )
        gene_ann = gene_ann.to_numpy().reshape(data.shape[1])
        if (len(gene_ann) != data.shape[1]):
        	raise ValueError("The length of the gene annotation list is not" +
              	          	 " equal to the number of genes !")
    else: 
        gene_ann = None
    data = data.to_numpy()
    labels = labels.to_numpy().reshape(data.shape[0])
    #- Create ScMags object
    obj = ScMags(
        data=data, 
        labels=labels,
        gene_ann=gene_ann,
        verbose=args['verbose']
    )
    #- Filter genes
    obj.filter_genes(
        in_cls_thres=args['in_cls_thres'],
        nof_sel=args['nof_sel'], 
        im_exp=args['im_exp'], 
        log_norm=args['log_norm']
    )
    #- Select Markers
    obj.sel_clust_marker(
        nof_markers = args['nof_markers'], 
        n_cores=args['n_cores'],
        dyn_prog = args['dyn_prog']
    ) 
    if args['dotplot']:
        obj.dot_plot()
    if args['markertsne']:
        obj.markers_tSNE()
    if args['markerheat']:
        obj.markers_heatmap()
    if args['knnmark']:
        obj.KNN_classifier()
    if args['write_res']:
        pth = args['data']
        if '/' in pth:
            dt_name = pth.split(sep='/')[-1]
            pth = pth[:len(pth)-len(dt_name)]
            ind_name = (pth + 
                        dt_name.split(sep = '.')[0] + '_markers_res_ind.csv')
            ann_name = (pth + 
                        dt_name.split(sep = '.')[0] + '_markers_res_ann.csv')
        else:
            dt_name = pth.split(sep='.')[0]
            ind_name = dt_name + '_markers_res_ind.csv'
            ann_name = dt_name + '_markers_res_ann.csv'
        
        mark_ind = obj.get_markers(ind_return=True)
        mark_ann = obj.get_markers(ind_return=False)
        
        mark_ind.to_csv(ind_name)
        mark_ann.to_csv(ann_name)
        
    if args['verbose']:
        print(obj.get_markers())
