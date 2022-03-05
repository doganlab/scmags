import os
import sys
import scipy
import numpy as np
import pandas as pd

from scipy import sparse
from joblib import Parallel, delayed
from typing import Optional, Union
from joblib.externals.loky import get_reusable_executor
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from ._plot import MarkerVis

class ScMags(MarkerVis):
    
    """\
    A class containing the Count matrix and cluster labels, with methods 
    that can select markers and visualize selected markers

    Parameters
    ----------
    data 
        Count matrix with rows corresponding to cells and columns to genes in 
        numpy array format.
    labels 
        One-dimensional numpy array containing cluster labels.
    gene_ann 
        One-dimensional numpy array containing gene annotations.
    verbose 
        Verbosity

    Returns
    -------
    :class: `~ScMags`.
    
    Examples
    --------
    >>> import scmags as mg
    >>> li = mg.datasets.li()
    """
    
    def __init__(            
        self,
        data: Union[np.ndarray, 
                    scipy.sparse.csr_matrix, 
                    scipy.sparse.csc_matrix],
        labels: Union[np.ndarray],
        gene_ann: Optional[np.ndarray] = None,
        verbose: bool = True   
    ):
        
        #- Initialize with visualization class parameters
        super().__init__()
        if gene_ann is None:
            gene_ann = list(np.arange(0, data.shape[1]).astype(str))
            gene_ann = ["Gene_" + i for i in gene_ann]
        if (len(labels) != data.shape[0]):
            raise ValueError("Label count and cell count are not equal !")
        if (len(gene_ann) != data.shape[1]):
            raise ValueError("The length of the gene annotation list is not" +
                             " equal to the number of genes !")
            
        self._sprs_form = True
        if(sparse.issparse(data) is False):
            self._sprs_form = False
        if(data.shape[0] < 1e5):
            self._silh = True
        else:
            self._silh = False
            
        self.data = data
        self.labels = labels
        self.gene_ann = gene_ann
        self._verbose = verbose
        self._markers = None
        self._sel_ind = None
        
#-------------------------------- Methods ------------------------------------# 
    
    @staticmethod
    def _analyze_genes(
        clstr_rnk: int,
        mean_rnk: int,
        nof_sel: int,
        im_exp: int,
        auto_thres_val: float,
        filt_proc: Union[np.ndarray],
        exp_sel: Union[np.ndarray],
        in_exp_rates: Union[np.ndarray],
        in_clst_means: Union[np.ndarray],
    ):
   
        # - Elimination of low-expression genes within the j Nth cluster
        high_exp = np.where(in_exp_rates[clstr_rnk, :] >= auto_thres_val)[0]
        
        # Returns None if there is no gene above the threshold
        if len(high_exp) == 0:
            return None
        
        filt_proc[1, exp_sel[high_exp]] = True
        temp_exp = in_exp_rates[:, high_exp]
        temp_mean = in_clst_means[:, high_exp]
        
        # Genes with the maximum mean within the cluster
        max_mean = np.argsort(temp_mean, axis=0)[::-1]
        # Here mean_rnk becomes 0 in the first while loop and 1 in the 
        # second while loop. So the maximum mean requirement is removed
        max_ind = np.where(max_mean[mean_rnk, :] == clstr_rnk)[0]
        # Returns None if there is no gene with the maximum within-cluster mean.
        if len(max_ind) == 0:
            return None
        max_mean = max_mean[:, max_ind]
        temp_ind = np.where(filt_proc[1, :] == True)[0]
        filt_proc[2, temp_ind[max_ind]] = True
        temp_mean = temp_mean[:, max_ind]
        temp_exp = temp_exp[:, max_ind]
        
        if mean_rnk == 0:
            thsec = 1
        else:
            thsec = 0
            
        # Selecting genes with high difference in within-cluster means
        diff = np.zeros(len(max_ind))
        div_exp = np.zeros(len(max_ind))
        mean_ind = np.arange(temp_exp.shape[0])
        mean_ind = np.delete(mean_ind, clstr_rnk)
        for k in range(len(max_ind)):
    
            div_exp[k] = np.mean(temp_exp[mean_ind, k])
            diff[k] = temp_mean[max_mean[mean_rnk,k], k] - temp_mean[max_mean[thsec,k], k]
    
        div_exp += 1e-9  # For divide error
        if len(diff) > 1:
            diff = (((diff - diff.min()) / (diff.max() - diff.min())) * 9) + 1
            
        # Product of within-cluster expression rates and difference between means
        diff = diff * pow(temp_exp[clstr_rnk, :], im_exp) 
        
        # Division by non-cluster expression rates
        diff = diff / pow(div_exp, im_exp)
        ind = np.argsort(diff)[::-1][:nof_sel]
        temp_ind = np.where(filt_proc[2, :] == True)[0]
        filt_proc[3, temp_ind[ind]] = True
        best_temp = np.where(filt_proc[3, :] == True)[0]
    
        # Returns marker indexes and scores
        return best_temp, diff[ind]
    
#-----------------------------------------------------------------------------#

    def filter_genes(
            self,
            in_cls_thres: Optional[float] = None,
            im_exp: Optional[int] = 10,
            nof_sel: Optional[int] = 10, 
            log_norm: bool = True
            
    ):   
        """
        It filters out genes that cannot be cluster-specific markers for each 
        cluster for computational efficiency. In this way, in the next step, 
        calculations are made for a specific gene community for each cluster.

        Parameters
        ----------
        in_cls_thres 
            Minimum expression threshold within the cluster. If it is not 
            given, it is computed automatically from the boxplot values. 
        im_exp 
            The importance of expression rates in clusters outside the 
            filtered cluster when performing gene filtering for any cluster.
            If it is increased, the expression rates in clusters other than 
            the filtered cluster will decrease.
        nof_sel 
            The number of genes remained for each cluster after filtering.
            Marker selection will be made on these genes.
        log_norm 
            Log normalization status.
            
        Example 
        -------
        >>> import scmags as mg
        Li dataset
        >>> li = mg.datasets.li()
        Filtering out unnecessary genes
        >>> li.filter_genes()
        
        After filtering genes you can see;
        
        .. hlist::
            :columns: 1
            
            * Remaining genes, 
            * Scores of the remaining genes, 
            * Threshold values determined for each cluster
        
        >>> rem_genes = li.get_filter_genes()
        >>> gen_scores = li.get_filter_gene_scores()
        >>> li.get_filt_cluster_thresholds
        """
        if (in_cls_thres is None):
            auto_thres = True
        else:
            auto_thres = False
            
        nof_cell, nof_gene = self.data.shape
        unq_labs = np.unique(self.labels)
        nof_clstr = len(unq_labs)  
        filt_proc = np.zeros((4, nof_gene), dtype=bool)
        
        if(self._verbose):
            print('-> Eliminating low expression genes')
            
        # Calculation of within cluster expression rates
        clst_index = []
        in_exp_rates = np.zeros((nof_clstr, nof_gene), dtype='f4')
        if self._sprs_form:
            for i in range(nof_clstr):
                clst_index.append(np.where(self.labels == unq_labs[i])[0])                
                temp = self.data[clst_index[i], :]
                in_exp_rates[i, :] = temp.getnnz(axis = 0) / temp.shape[0]
        else:
            for i in range(nof_clstr):
                clst_index.append(np.where(self.labels == unq_labs[i])[0])                
                temp = self.data[clst_index[i], :]
                in_exp_rates[i, :] = ((temp != 0).sum(axis=0))
                in_exp_rates[i, :] = in_exp_rates[i, :] / temp.shape[0]
                
        # Genes with less than 20% expression in all clusters are filtered out.
        threshed_exp = np.sum((in_exp_rates < 0.2), axis = 0)
        high_exp_ind = np.where(threshed_exp < in_exp_rates.shape[0])[0]
        in_exp_rates = in_exp_rates[:, high_exp_ind]
        filt_proc[0, high_exp_ind] = True  
        
        # Calculation of within cluster means 
        in_clst_means = np.zeros((nof_clstr, len(high_exp_ind)), dtype='f4')
        if self._sprs_form:
            for i in range(nof_clstr):
                logdata = self.data[clst_index[i]][:, high_exp_ind].copy()
                logdata.data = np.log2(logdata.data + 1, dtype='f4')
                in_clst_means[i, :] = np.mean(logdata, axis=0).A[0] 
        else:
            for i in range(nof_clstr):
                logdata = self.data[clst_index[i]][:, high_exp_ind].copy()
                logdata = np.log2(logdata + 1, dtype='f4')
                in_clst_means[i, :] = np.mean(logdata, axis=0)
                        
        if(self._verbose): 
            print('-> Selecting cluster-specific candidate marker genes')
            
        sel_ind = {}
        sel_score = {}
        clst_thres = {}
        exp_sel = np.where(filt_proc[0, :] == True)[0]
        for j in range(nof_clstr):
            # Determination of in-cluster expression thresholds
            if auto_thres:
                iter_thres = 10
                desc_temp = in_exp_rates[j, :]
                desc_temp = pd.DataFrame(desc_temp[desc_temp != 0])
                auto_thres_val = (desc_temp.describe().iloc[6, 0] + 
                                  desc_temp.describe().iloc[7, 0]) / 2
            else:
                iter_thres = 1
                auto_thres_val = in_cls_thres
        
            sel_len = 0
            nof_iter = 0
            mean_rnk = 0
            clst_thres[unq_labs[j]] = auto_thres_val
            # It is aimed to select genes that are above the determined 
            # expression rate and have the maximum within-cluster mean.
            # If there are not as many genes as nof_sel, genes that exceed the 
            # threshold are selected. The threshold is lowered and the 
            # processes are repeated to complete the number.
            while (sel_len < nof_sel) and (nof_iter < iter_thres):
                nof_iter += 1
                best_genes = self._analyze_genes(
                    clstr_rnk=j,
                    mean_rnk=mean_rnk,
                    nof_sel=nof_sel,
                    im_exp=im_exp,
                    auto_thres_val=auto_thres_val,
                    filt_proc=filt_proc,
                    exp_sel=exp_sel,
                    in_exp_rates=in_exp_rates,
                    in_clst_means=in_clst_means
                )
                if best_genes is not None:
                    sel_len = len(best_genes[0])
                auto_thres_val -= auto_thres_val * 0.01
                filt_proc[1:4, :] = False
        
            if best_genes is not None:
                sel_ind[unq_labs[j]] = best_genes[0]
                sel_score[unq_labs[j]] = best_genes[1]
            
            # If no gene is obtained from the previous loop, it enters 
            # this loop.
            rem_sel_len = nof_sel - sel_len
            sel_len = 0
            while (sel_len < rem_sel_len) and (mean_rnk < nof_clstr-1):
        
                mean_rnk += 1
                best_genes = self._analyze_genes(
                    clstr_rnk=j,
                    mean_rnk=mean_rnk,
                    nof_sel=rem_sel_len,
                    im_exp=im_exp,
                    auto_thres_val=clst_thres[unq_labs[j]],
                    filt_proc=filt_proc,
                    exp_sel=exp_sel,
                    in_exp_rates=in_exp_rates,
                    in_clst_means=in_clst_means
                )
                
                filt_proc[1:4, :] = False
                if best_genes is not None:
                    sel_len = len(best_genes[0])
                    rem_sel_len = rem_sel_len - sel_len 
                    if unq_labs[j] in list(sel_ind.keys()):
                        sel_ind[unq_labs[j]] = np.append(
                            sel_ind[unq_labs[j]], best_genes[0])
                        sel_score[unq_labs[j]] = np.append(
                            sel_score[unq_labs[j]], best_genes[1])
                    else:
                        sel_ind[unq_labs[j]] = best_genes[0]
                        sel_score[unq_labs[j]] = best_genes[1]
            
            if nof_iter > 1:
                clst_thres[unq_labs[j]] = auto_thres_val

        self._sel_ind = sel_ind
        self._sel_score = sel_score
        self._clst_thres = clst_thres
#-----------------------------------------------------------------------------#

    def silh_compute(self, data, mask_labs, i, j):
        """
        Calculates silhouette index of any gene
        """
        if self._silh:
            score = silhouette_score(data[:, j], mask_labs[i])
        else:
            score = calinski_harabasz_score(data[:, j].toarray(), mask_labs[i])

        return score
   
#-----------------------------------------------------------------------------#

    def dp_silh_compute(self, data, mask, pb_it):
        """
        It finds the best combination of given genes by parallel computation 
        based on silhouette index.
        
        """
        #- Number of genes to calculate
        n_cores = self._n_cores
        nof_markers = self._nof_markers
        nof_gene = data.shape[1]  
        
        #- Number of available cores
        dtc_cores = os.cpu_count()
        
        #- If the given number of cores is greater than the 
        # current number of cores
        if n_cores > dtc_cores:
            n_cores = dtc_cores
            
        #- Setting the number of cores to use
        if(n_cores == -2 and dtc_cores >= nof_gene):
            tmp_cores = nof_gene
        elif(n_cores == -2 and dtc_cores < nof_gene):
            tmp_cores = dtc_cores    
        elif(n_cores != -2 and n_cores >= nof_gene):
            tmp_cores = nof_gene
        elif(n_cores != -2 and n_cores < nof_gene):
            tmp_cores = n_cores
            
        whole_ınd = list(range(nof_gene))
        score_change = np.zeros(nof_markers, dtype="f4")
        cmb = whole_ınd
        if self._silh:
            cmb_res = Parallel(n_jobs=tmp_cores, backend="loky")(
                delayed(silhouette_score)(data[:, j], mask) for j in cmb
            )
        else:
            cmb_res = Parallel(n_jobs=tmp_cores, backend="loky")(
                delayed(calinski_harabasz_score)(data[:, j], mask) for j in cmb
            )
            
        max_ınd = np.argmax(cmb_res)
        del whole_ınd[max_ınd]
        cmb = np.stack((np.repeat(max_ınd, nof_gene - 1), whole_ınd), axis=1)  
        nof_iter = min(nof_gene, nof_markers)  
        
        for k in range(1, nof_iter):
            
            if self._silh:               
                cmb_res = Parallel(n_jobs=tmp_cores, backend="loky")(
                    delayed(silhouette_score)(data[:, j], mask) for j in cmb
                )
            else:
                cmb_res = Parallel(n_jobs=tmp_cores, backend="loky")(
                    delayed(calinski_harabasz_score)(data[:, j], mask) 
                    for j in cmb
                )
                
            # Best Marker Index
            max_ınd = np.argmax(cmb_res)
            best_mark = cmb[max_ınd]
            score_change[k] = cmb_res[max_ınd]
            whole_ınd = np.setdiff1d(whole_ınd, cmb[max_ınd][-1])
            cmb = np.repeat([best_mark], len(whole_ınd), axis=0)
            
            # Creating combinations from remaining genes
            cmb = np.append(cmb, whole_ınd[:, np.newaxis], axis=1)
            
        if(self._verbose): 
            self._pb(self._tot_iter, pb_it+1)

        return best_mark
                
#-----------------------------------------------------------------------------#

    def sel_clust_marker(
            self,
            nof_markers: Optional[int] = 3,              
            n_cores: Optional[int] = -2,
            dyn_prog: bool = False
    ):
        """
        Performs cluster-specific marker selection among filtered genes. 
        It does this in two different ways.

        Parameters
        ----------
        nof_markers 
            Number of markers to be selected for each cluster. 
        n_cores 
            Number of cores to use. (If not given = total number of cores -1)
        dyn_prog 
            Marker selection status with dynamic programming. If True, the 
            combinational genes are combined and the gene combination with 
            the highest silhouette value is selected.
            
        Examples
        --------
        >>> import scmags as mg
        Li Dataset
        >>> li = mg.datasets.li()
        Filtering out unnecessary genes
        >>> li.filter_genes()
        Selection of markers from remaining genes
        >>> li.sel_clust_marker()
        Get markers 
        >>> li.get_markers()
        Get markers data
        >>> li.get_marker_data()
        """
        if(self._sel_ind is None):
            raise ValueError(
                'Marker selection cannot be made without gene filtering !'
                )
            
        self._n_cores = n_cores
        self._nof_markers = nof_markers
            
        labels = self.labels
        sel_ind = self._sel_ind.copy()
        
        #- Number of remaining genes after filtering for each cluster
        filt_counts = np.array([len(sel_ind[ky]) for ky in sel_ind.keys()])
        
        #- Indexes of remaining genes after filtering for each cluster
        filt_clst_ind = np.array(list(sel_ind.values()), dtype=object)
        mark_keys = np.array(list(sel_ind.keys()))
        nof_iter = len(mark_keys)  

        self._tot_iter = nof_iter
        self._mark_keys = mark_keys

        #- Log normalization for computation of remaining genes after filtering
        clst_logdata = []
        if self._sprs_form:
            for i in range(nof_iter):
                 clst_logdata.append(
                     self.data[:, filt_clst_ind[i].astype(int)].copy())
                 clst_logdata[i].data = np.log2(
                     (clst_logdata[i].data + 1), dtype='f4')
        else:
            for i in range(nof_iter):
                clst_logdata.append(
                    sparse.csr_matrix(
                        self.data[:, filt_clst_ind[i].astype(int)].copy()))
                clst_logdata[i].data = np.log2(
                    (clst_logdata[i].data + 1), dtype='f4')
                
        #- Clıster specific masks
        mask_labs = []
        for k in range(nof_iter):     
            all_same = np.zeros(len(labels), dtype=int)
            all_same[labels == mark_keys[k]] = 1  
            mask_labs.append(all_same)
            
        if(self._verbose): 
            print('-> Selecting  markers for each cluster')
            
        if dyn_prog:
            sel_mark = []
            for i in range(nof_iter):
                res = self.dp_silh_compute(
                    data=clst_logdata[i], mask=mask_labs[i], pb_it=i
                )
                sel_mark.append(filt_clst_ind[i][res])
            get_reusable_executor().shutdown(wait=False)
        else:
            res = Parallel(n_jobs=n_cores, backend="loky")(
                delayed(self.silh_compute)(
                    data=clst_logdata[i], mask_labs=mask_labs, i=i, j=j
                )
                for i in range(nof_iter)
                for j in range(filt_counts[i])
            )
            
            get_reusable_executor().shutdown(wait=False)
            sel_mark = []
            for k in range(nof_iter):
                temp = res[:filt_counts[k]]
                del res[:filt_counts[k]]
                first_n = min(filt_counts[k], nof_markers)
                srt = np.argsort(temp)[::-1][:first_n]
                sel_mark.append(filt_clst_ind[k][srt])  
        
        marker_df = pd.DataFrame(sel_mark)
        marker_df.index = ["C_" + str(i) for i in mark_keys]  
        marker_df.columns = [
            "Marker_" + str(i) for i in range(1, nof_markers + 1)
        ]        
        self._sel_list_markers = sel_mark
        self._markers = marker_df
        
#-----------------------------------------------------------------------------#        
    def _pb(self, total, progress):
        
        barLength = 20
        status = ""
        step = progress
        progress = float(progress) / float(total)
        if progress >= 1.0:
            progress = 1
            status = "\r\n"
        block = int(round(barLength * progress))
        
        text = (
            "\r-> |{}| {:.0f}% Number of Clusters With "
            + "Selected Markers : {}  {}"
        ).format(
            u"\u2B1B" * block + u"\u2B1C" * (barLength - block),
            round(progress * 100, 0),
            step,
            status,
        )
        
        sys.stdout.write(text)
        sys.stdout.flush()
            
#- Source: https://stackoverflow.com/questions/3160699/python-progress-bar

#--------------------------- Getter-Setter -----------------------------------#
 
    def get_filter_genes(self, ind_return: bool = False):
        """
        Returns the remaining genes after filtering.

        Parameters
        ----------
        ind_return 
            If True it returns the indices, if False it returns the 
            annotation if there is an gene annotation.

        Returns
        -------
        Dictionary
            Dictionary containing the remaining genes after filtering.
        """
        sel_ind = self._sel_ind.copy()
        gene_ann = self.gene_ann
        if ind_return:
            return sel_ind
        else:
            sel_ann = {}
            for i in sel_ind.keys():
                sel_ann[i] = []
                for j in range(len(sel_ind[i])):
                   sel_ann[i].append(gene_ann[sel_ind[i][j]])
            return sel_ann
        
    def get_markers(self, ind_return: bool = False): 
        """
        Returns the selected markers

        Parameters
        ----------
        ind_return 
            If True it returns the indices, if False it returns the 
            annotation if there is an gene annotation.

        Returns
        -------
        pd.DataFrame
            Dataframe containing indices or names of selected markers.

        """
        if ind_return:
            return self._markers.copy()
        else:
            named_df = self._markers.copy()
            for i in range(named_df.shape[0]):
                not_nan = np.where(named_df.iloc[i, :].isna() == False)[0]
                for j in not_nan:
                    named_df.iloc[i, j] = self.gene_ann[int(named_df.iloc[i, j])]
            return named_df
        
    def get_marker_data(self, log_norm: bool = True):
        """
        It pulls the selected markers from the count matrix and returns it.

        Parameters
        ----------
        log_norm : 
            Log normalization status. If true it normalizes and returns.

        """
        sel_ind = self._sel_list_markers.copy()
        sel_key = list(self._sel_ind.keys())
        marker_data = {}
        for i in range(len(sel_key)):
            marker_data[sel_key[i]] = self.data[:, sel_ind[i].astype(int)]
                
        if log_norm:
            if self._sprs_form:
                for i in sel_key:
                    marker_data[i].data = np.log2(marker_data[i].data + 1)
            else:
                for i in sel_key:
                    marker_data[i] = np.log2(marker_data[i] + 1)
                
        return marker_data
    
    @property
    def get_filter_gene_scores(self):
        """
        Returns the calculated score values of the remaining genes after 
        filtering.
        """
        return self._sel_score.copy()
    
    @property
    def get_filt_cluster_thresholds(self):
        """
        Returns auto-determined in-cluster expression threshold values
        """
        thres = self._clst_thres.copy()
        if isinstance(thres, dict):
            return thres
        else:
            print("Threshold was defined as %.2f for all clusters" % thres)
            
#-----------------------------------------------------------------------------#