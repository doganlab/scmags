import os 
import numpy as np
import matplotlib.colorbar
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import cm
from matplotlib import colors
from matplotlib.path import Path
from matplotlib.axes import Subplot  
from matplotlib.colors import Normalize
from matplotlib.figure import SubplotParams as Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, GridSpecBase
from typing import Union, Sequence, Optional, Tuple, Literal

class MarkerVis():
    
    """
    Class with methods that can visualize selected markers
    """
#-----------------------------------------------------------------------------#
    _label_colors = [
        "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", 
        "#A30059", "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", 
        "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693", "#FEFFE6", 
        "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", 
        "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", 
        "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", 
        "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500", "#C2FFED",
        "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C", 
        "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", 
        "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", 
        "#456648", "#0086ED", "#886F4C", "#34362D", "#B4A8BD", "#00A6AA", 
        "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81", "#575329", 
        "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", 
        "#1E6E00", "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", 
        "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F", "#201625", 
        "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", 
        "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", 
        "#6A3A4C", "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", 
        "#A3A489", "#806C66", "#222800", "#BF5650", "#E83000", "#66796D", 
        "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51", "#C895C5", 
        "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", 
        "#012C58"
     ]

    _label_map = colors.LinearSegmentedColormap.from_list(
        'mymap', _label_colors
        )
#-----------------------------------------------------------------------------#

    # tSNE Defaults
    _DEFAULT_TSNE_CMAP = _label_map
    _DEFAULT_TSNE_FSIZE = (13,11)
    _DEFAULT_TSNE_TITLE = 'tSNE Plot of Markers'
    _DEFAULT_TSNE_NAME = "scmags_tsne_of_markers"

    # Confusion Matrix Defaults
    _DEFAULT_CONF_CMAP = cm.get_cmap('gist_heat_r')
    _DEFAULT_CONF_NAME = "scmags_knn_result"
    
    # Heatmap Defaults
    _DEFAULT_HEAT_FSIZE = (15,16)
    _DEFAULT_HEAT_CMAP = cm.get_cmap('YlGnBu')
    _DEFAULT_HEAT_NAME = "scmags_heatmap_of_markers"
    
    #- Dotplot Defaults
    _DEFAULT_WSPACE = 0
    _DEFAULT_FIG_SIZE = (6,14)
    _DEFAULT_LEGENDS_WIDTH = 1.5
    _DEFAULT_BRACKET_WIDTH = 0.35
    _DEFAULT_CMAP = cm.get_cmap('OrRd')
    _DEFAULT_DOT_MAX = None
    _DEFAULT_DOT_MIN = None
    _DEFAULT_LARGEST_DOT = 90.0
    _DEFAULT_SMALLEST_DOT = 0.0
    _DEFAULT_DOT_EDGELW = 1.2
    _DEFAULT_DOT_EDGECOLOR = 'k'
    _DEFAULT_SIZE_EXPONENT = 1.5
    _DEFAULT_TOP_ADJUSTMENT = -0.5
    _DEFAULT_BOT_ADJUSTMENT = 0.5
    _DEFAULT_SIZE_LEGEND_TITLE = 'Fraction of cells\nin group (%)'
    _DEFAULT_COLOR_LEGEND_TITLE = 'Mean expression\nin group'
    _DEFAULT_DOT_NAME = "scmags_dotplot"
    
    def __init__(self):
        
        self._wspace = self._DEFAULT_WSPACE
        self._figsize = self._DEFAULT_FIG_SIZE
        self._legends_width = self._DEFAULT_LEGENDS_WIDTH 
        self._bracket_width = self._DEFAULT_BRACKET_WIDTH 
        self._cmap = self._DEFAULT_CMAP
        self._dot_max = self._DEFAULT_DOT_MAX 
        self._dot_min = self._DEFAULT_DOT_MIN 
        self._sml_dot = self._DEFAULT_SMALLEST_DOT
        self._lrg_dot = self._DEFAULT_LARGEST_DOT 
        self._dot_edge_color = self._DEFAULT_DOT_EDGECOLOR 
        self._dot_edge_lw = self._DEFAULT_DOT_EDGELW 
        self._size_exp = self._DEFAULT_SIZE_EXPONENT 
        self._top_adj = self._DEFAULT_TOP_ADJUSTMENT
        self._bot_adj = self._DEFAULT_BOT_ADJUSTMENT
        self._size_leg_title = self._DEFAULT_SIZE_LEGEND_TITLE 
        self._col_leg_title = self._DEFAULT_COLOR_LEGEND_TITLE 
        
#-----------------------------------------------------------------------------#

    def markers_heatmap(
            self,
            scale: bool = True,
            log_norm: bool = True,
            cmap: Optional[str] = _DEFAULT_HEAT_CMAP,
            figsize: Optional[Tuple[int, int]] = _DEFAULT_HEAT_FSIZE,
            plot_name: Optional[str] = _DEFAULT_HEAT_NAME,
            save_plot: bool = False,
            plot_dpi: int = 300
    ):
        
        """
        Draws a heatmap with selected markers
        
        Parameters
        ----------
        scale 
            Scales marker genes from 0 to 1. 
        log_norm 
            Performs log normalization before scaling.
        figsize :
            Figure size.
        cmap : 
            A Colormap instance or registered colormap name. (matplotlib cmap)
        plot_name
            Name of plot to save
        save_plot
            If True, the plot is saved in the working directory.
        plot_dpi
            The dpi value of the figure to be saved
            
        Examples
        --------
        .. plot::
           :include-source: True
        
           >>> import scmags as mg
           >>> li = mg.datasets.li()
           >>> li.filter_genes()
           >>> li.sel_clust_marker()
           >>> li.markers_heatmap(figsize = (13,15))

        """
        if cmap != self._DEFAULT_HEAT_CMAP or isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
        nc = self._tot_iter
        labels = self.labels
        unq_labs = self._mark_keys
        
        # Markers
        sel_mark_ind = self._sel_list_markers.copy()
        iters = [len(sel_mark_ind[ky]) for ky in range(len(sel_mark_ind))]
        min_mark = min(iters)
        self._min_mark = min_mark
        
        # Arranges elements from the same set one after the other
        srt_clst = []
        for j in range(nc):
            srt_clst.append(np.where(labels == unq_labs[j])[0])
            
        srt_clst = np.hstack(srt_clst)
        srt_labs = labels[srt_clst]
        
        # The smallest number of markers selected
        cluster_markers = np.concatenate(
            np.array(self._markers.iloc[:,:min_mark]))  
        
        # Data with selected markers and sorted elements 
        logdata = self.data[:, cluster_markers.astype(int)].copy()
        logdata = logdata[srt_clst, :]
        
        # Log normalization 
        if log_norm:
            if self._sprs_form:
                logdata.data = np.log2(logdata.data + 1, dtype='f4')
            else:
                logdata = np.log2(logdata + 1, dtype='f4')
                
        #- Gene Based Scaling
        if scale:
            if self._sprs_form:
                scaler = MaxAbsScaler()
                scaled = scaler.fit_transform(logdata)
                scaled = scaled[::-1].toarray()
            else:
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(logdata)
                scaled = scaled[::-1]
        else:
            if self._sprs_form:
                scaled = logdata[::-1].toarray()
            else:
                scaled = logdata[::-1]
        
        #- If the number of clusters is more than 25, 2 lines are set 
        # for the legends.
        if len(unq_labs) < 25:
            width_ratios = [0.85, 0.15]
            wspace = 0.15
            ncol = 1
        else:
            width_ratios = [0.65, 0.35]
            wspace = 0.20
            ncol = 2

        fig, gs = self._make_grid_spec(
            ax_or_figsize=figsize,
            nrows=3,
            ncols=2,
            wspace=wspace,
            hspace=0.15,
            height_ratios=[0.05, 0.9, 0.05],
            width_ratios=width_ratios
        )
        
        mainplot_gs = GridSpecFromSubplotSpec(
            nrows=2,
            ncols=2,
            wspace=0,
            hspace=0,
            subplot_spec=gs[1, 0],
            width_ratios=[0.96,0.04],
            height_ratios=[0.97,0.03],
        )
        
        cbar_ax = fig.add_subplot(gs[0,1])
        clst_leg_ax = fig.add_subplot(gs[1:3,1])
        heat_ax = fig.add_subplot(mainplot_gs[0,0])
        brac_ax = fig.add_subplot(mainplot_gs[1,0])
        leg_col_ax = fig.add_subplot(mainplot_gs[0,1])
        tick_kw = {
            'top': False, 'bottom': False, 
            'left': False, 'right': False,  
            'labeltop': False, 'labelbottom': False,
            'labelright': False, 'labelleft': False
        }
        
        #- Make Heatmap
        heat_ax.pcolor(scaled,  cmap=cmap)
        heat_ax.tick_params(
            axis='both',
            which='both',
            top=True,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
            labeltop=True,
            labelbottom=False,
            labelsize="x-large"
        )
        
        # Add Gene annotations 
        x_ticks = np.arange(scaled.shape[1]) + 0.5
        heat_ax.set_xticks(x_ticks)
        heat_ax.set_xticklabels(
            [self.gene_ann[idx] for idx in cluster_markers],
            rotation=90,
            ha='center',
            minor=False,
        )
        for axis in ['top', 'bottom','left']:
            heat_ax.spines[axis].set_linewidth(1.8)
            
        # Plot brackets    
        self._plot_bracket(brac_ax, heat_brac=True)
        
        # Add Colorbar 
        c_norm = colors.Normalize(vmin=None, vmax=None)
        matplotlib.colorbar.ColorbarBase(
            cbar_ax, 
            orientation='horizontal', 
            cmap=cmap, 
            norm=c_norm
        )
        cbar_ax.set_title("Normalized Expression", fontsize='medium')
        cbar_ax.xaxis.set_tick_params(labelsize='small')
        
        if nc < 12 :
            leg_pal = cm.get_cmap('Paired').colors
            lab_pal = colors.LinearSegmentedColormap.from_list(
                'labpal', leg_pal[:nc]
                )
        else:
            leg_pal = self._label_colors
            lab_pal = colors.LinearSegmentedColormap.from_list(
                'labpal', leg_pal[:nc]
                )

        # Legend Handles
        handles = []
        for i in range(nc):
            patch = patches.Patch(color=leg_pal[i], label=unq_labs[i])
            handles.append(patch) 
        
        # Plot the legends
        clst_leg_ax.legend(
            handles=handles, 
            labels=list(unq_labs),
            loc='upper center',
            fancybox=True,
            shadow=True,
            markerscale=1.2,
            fontsize='x-large',
            ncol=ncol,
            title='Label Colors',
            title_fontsize='x-large'
        )
        clst_leg_ax.tick_params(axis='both', which='both', **tick_kw)
        for axis in ['top', 'bottom', 'left', 'right']:
            clst_leg_ax.spines[axis].set_visible(False)
        clst_leg_ax.grid(False)
        
        # Label Colors 
        le = LabelEncoder()
        le.fit(srt_labs)
        labs_ınt = le.transform(srt_labs)
        labs_ınt = labs_ınt.reshape(len(labs_ınt),1)[::-1]
        leg_col_ax.pcolor(labs_ınt, cmap=lab_pal)
        leg_col_ax.tick_params(axis='both', which='both', **tick_kw)
        for axis in ['top', 'bottom',  'right']:
            leg_col_ax.spines[axis].set_linewidth(1.5)
          
        plt.show() 
        if save_plot:
            plot_path = os.getcwd() + '/' + plot_name + '.png'
            fig.savefig(plot_path, dpi=plot_dpi)
#-----------------------------------------------------------------------------#

    def knn_classifier(
            self, 
            test_ratio: Optional[float] = 0.3, 
            nof_neighbors: Optional[int] = 3,
            log_norm: Optional[bool] = True,
            conf_normalize: Optional[Literal['true', 'pred', 'all']] = 'true',
            rand_state: Optional[int] = None,
            main_title: Optional[str] = "Confusion Matrix",
            figsize: Optional[Tuple[int, int]] = None,
            cmap: Optional[str] = _DEFAULT_CONF_CMAP,
            plot_name: Optional[str] = _DEFAULT_CONF_NAME,
            save_plot: bool = False,
            plot_dpi: int = 300
    ):
        """
        This function performs k-NN classification with selected markers. 
        Visualizes the results with a normalized confusion matrix.
        
        Parameters
        ----------
        test_ratio 
            Test data rate. Should be between 0-1. 
        nof_neighbors 
            Number of neighbors to use. 
        log_norm 
            Performs log normalization before k-NN. 
        conf_normalize 
            Normalizes confusion matrix over the true (rows), predicted 
            (columns) conditions or all the population. If None, confusion 
            matrix will not be normalized. 
        rand_state 
            Controls the shuffling applied to the data before applying the 
            split.
        main_title
            Main title for confusion matrix. 
        figsize
            Figure size. If not given its own settings.
        cmap : Optional[str], optional
            A Colormap instance or registered colormap name. (matplotlib cmap)
        plot_name
            Name of plot to save
        save_plot
            If True, the plot is saved in the working directory.
        plot_dpi
            The dpi value of the figure to be saved
            
        Examples
        --------
        .. plot::
           :include-source: True
        
           >>> import scmags as mg
           >>> li = mg.datasets.li()
           >>> li.filter_genes()
           >>> li.sel_clust_marker()
           >>> li.knn_classifier()
        
        """
        if cmap != self._DEFAULT_CONF_CMAP or isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
        labels = self.labels
        cluster_markers = np.hstack(self._sel_list_markers)
        sel_data = self.data[:,cluster_markers.astype(int)].copy() 

        le = LabelEncoder()
        le.fit(labels)
        labs_ınt = le.transform(labels)
        
        # Log Normalization
        if log_norm:
            if self._sprs_form:
                sel_data.data = np.log2(sel_data.data + 1, dtype='f4')
            else:
                sel_data = np.log2(sel_data + 1, dtype='f4')
                
        # Test train split
        split_res = train_test_split(
            sel_data, 
            labs_ınt, 
            test_size=test_ratio,
            random_state=rand_state
        )
        
        # k-NN classification
        knn_model = KNeighborsClassifier(n_neighbors=nof_neighbors)
        knn_model.fit(split_res[0], split_res[2])  
        test_preds = knn_model.predict(split_res[1])
        conf_mat = confusion_matrix(
            y_true=split_res[3], 
            y_pred=test_preds, 
            labels=np.unique(split_res[3]),
            normalize=conf_normalize
        )
        inv_test = le.inverse_transform(np.unique(split_res[3]))
        
        # Set figure size
        if figsize is None:
            sz = len(knn_model.classes_)
            if sz > 20: sz = 20
            if sz < 12: sz = 12 
            figsize = (sz,sz+1)
        fig, gs = self._make_grid_spec(
            ax_or_figsize=figsize,
            nrows=5,
            ncols=2,
            wspace=0.04,
            width_ratios=[0.95, 0.05],
            height_ratios=[0.05,0.1, 0.8, 0.1,0.05]
        )
        
        # Main confusion matrix ax
        conf_ax = fig.add_subplot(gs[:,0])
        #-Legend ax
        color_legend_ax = fig.add_subplot(gs[2,1])
        for axis in ['top', 'bottom', 'left', 'right']:
                conf_ax.spines[axis].set_linewidth(1.5)
        conf_ax.matshow(conf_mat, cmap=cmap)
        for (i, j), z in np.ndenumerate(conf_mat):
            mp = 0.99
            if (conf_mat[i,j] >= 0.5):
                mp = 0.01
            conf_ax.text(
                j, i, 
                s='{:0.2f}'.format(z), 
                ha='center', 
                va='center',
                size='large',
                weight='roman',
                color=cmap(mp)
            )
       
        for axis in ['top', 'bottom', 'left', 'right']:
                    conf_ax.spines[axis].set_linewidth(1.5)
                    
        # Set x ticks
        x_ticks = np.arange(conf_mat.shape[0]) 
        conf_ax.set_xticks(x_ticks)
        conf_ax.set_xticklabels(inv_test, rotation=90)
        
        # Set y ticks        
        y_ticks = np.arange(conf_mat.shape[0]) 
        conf_ax.set_yticks(y_ticks, minor=False)
        conf_ax.set_yticklabels(inv_test)
        conf_ax.tick_params(axis='both', labelsize='large')
        conf_ax.xaxis.tick_bottom()
        
        # Set title and labels
        conf_ax.set_title(main_title, size="xx-large", weight="bold", pad=30)
        conf_ax.set_ylabel(
            'Actual Labels', size="x-large", weight='semibold')
        conf_ax.set_xlabel(
            'Predicted Labels', size="x-large", weight='semibold')
        
        # Add Colorbar
        c_norm = colors.Normalize(vmin=None, vmax=None)
        matplotlib.colorbar.ColorbarBase(
            ax=color_legend_ax, 
            orientation='vertical', 
            cmap=cmap, 
            norm=c_norm
        )
        color_legend_ax.xaxis.set_tick_params(labelsize='small')
        
        plt.show()
        if save_plot:
            plot_path = os.getcwd() + '/' + plot_name + '.svg'
            fig.savefig(plot_path, dpi=plot_dpi)
#-----------------------------------------------------------------------------#

    def markers_tSNE(
            self, 
            log_norm: bool = True,
            n_iter: Optional[int] = 1000,
            perplexity: Optional[int] = 30,
            learning_rate: Optional[int] = 200,
            x_y_labels: bool = True,
            ax_spines: bool = True,
            main_title: Optional[str] = _DEFAULT_TSNE_TITLE,
            figsize: Optional[Tuple[int, int]] = _DEFAULT_TSNE_FSIZE,
            cmap: Optional[str] = _DEFAULT_TSNE_CMAP,
            plot_name: Optional[str] = _DEFAULT_TSNE_NAME,
            save_plot: bool = False,
            plot_dpi: int = 300
        ):
        
        """
        Draws tSNE plot with selected markers

        Parameters
        ----------
        log_norm 
            Performs log normalization before t-SNE. 
        n_iter 
            Maximum number of iterations for the optimization.
        perplexity 
            The perplexity is related to the number of nearest neighbors that 
            is used in other manifold learning algorithms. Larger datasets 
            usually require a larger perplexity.
        learning_rate 
            The learning rate for t-SNE is usually in the range [10.0, 1000.0].
        x_y_labels 
            Draws the x and y axis labels.
        ax_spines 
            Draws plot frames. 
        main_title : Optional[str], optional
            Main title..
        figsize 
            Figure size. 
        cmap 
            A Colormap instance or registered colormap name. (matplotlib cmap)
        plot_name
            Name of plot to save
        save_plot
            If True, the plot is saved in the working directory.
        plot_dpi
            The dpi value of the figure to be saved
            
        Examples
        --------
        .. plot::
           :include-source: True

           >>> import scmags as mg
           >>> li = mg.datasets.li()
           >>> li.filter_genes()
           >>> li.sel_clust_marker()
           >>> li.markers_tSNE()

        """
        
        from sklearn.manifold import TSNE
        
        if cmap != self._DEFAULT_TSNE_CMAP or isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
            
        # Data with selected markers
        cluster_markers = np.hstack(self._sel_list_markers)
        sel_data = self.data[:,cluster_markers.astype(int)].copy() 
        
        # Log Normalization
        if log_norm:
            if self._sprs_form:
                sel_data.data = np.log2(sel_data.data + 1 ,dtype='f4')
            else:
                sel_data = np.log2(sel_data + 1, dtype='f4')

        # Perform t-SNE
        sel_embedded = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter).fit_transform(sel_data)
        
        labels = self.labels
        le = LabelEncoder()
        le.fit(labels)
        labs_ınt = le.transform(labels)
        unq_labs = list(le.inverse_transform(np.unique(labs_ınt)))
        
        ncol = 1
        width_ratios = [0.85, 0.15]
        if len(unq_labs) > 25:
            width_ratios = [0.65, 0.35]
            ncol = 2
            
        # Create grid
        fig, gs = self._make_grid_spec(
            ax_or_figsize=figsize,
            nrows=1,
            ncols=2,
            wspace=0.22,
            width_ratios=width_ratios
        )
        
        if len(unq_labs) < 20:
            cmap = cm.get_cmap('tab20')
        tsne_ax = fig.add_subplot(gs[0,0])
        label_legend_ax = fig.add_subplot(gs[0,1])
        scatter = tsne_ax.scatter(
            sel_embedded[:,0], 
            sel_embedded[:,1], 
            c=labs_ınt,
            cmap=cmap
        )
        tsne_ax.tick_params(
            axis='both',
            which='both',
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False
        )
        
        # Add x-y labels
        if x_y_labels:
            tsne_ax.set_title(main_title, size="xx-large", weight='semibold')
            tsne_ax.set_xlabel("t-SNE Dim 1", size='x-large', style='oblique')
            tsne_ax.set_ylabel("t-SNE Dim 2", size='x-large', style='oblique')
            
        if not ax_spines:
            for axis in ['top', 'bottom', 'left', 'right']:
                tsne_ax.spines[axis].set_visible(False)
        
        # Add legends
        handles = scatter.legend_elements(num = len(unq_labs), alpha = 1)[0]
        label_legend_ax.legend(
            handles=handles, 
            labels=unq_labs,
            loc='upper center',
            fancybox=True,
            shadow=True,
            markerscale=1.5,
            fontsize='x-large',
            ncol=ncol,
            title='Label Colors',
            title_fontsize='x-large'
        )
        
        label_legend_ax.tick_params(
            axis='both',
            which='both',
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            labelbottom=False
        )
        for axis in ['top', 'bottom', 'left', 'right']:
            label_legend_ax.spines[axis].set_visible(False)
        
        plt.show()
        if save_plot:
            plot_path = os.getcwd() + '/' + plot_name + '.svg'
            fig.savefig(plot_path, dpi=plot_dpi)
#-----------------------------------------------------------------------------#

# In this section, scanpy library was used.

    @staticmethod
    def _make_grid_spec(
        ax_or_figsize: Union[Tuple[int, int], Subplot],
        nrows: int,
        ncols: int,
        wspace: Optional[float] = None,
        hspace: Optional[float] = None,
        width_ratios: Optional[Sequence[float]] = None,
        height_ratios: Optional[Sequence[float]] = None,
    ) -> Tuple[Figure, GridSpecBase]:
        
        kw = dict(
            wspace=wspace,
            hspace=hspace,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        if isinstance(ax_or_figsize, tuple):
            fig = plt.figure(figsize=ax_or_figsize)
            return fig, GridSpec(nrows, ncols, **kw)
        elif isinstance(ax_or_figsize, Subplot):
            ax = ax_or_figsize
            ax.axis('off')
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])
            return ax.figure, ax.get_subplotspec().subgridspec(
                                                    nrows, ncols, **kw)
        else: 
            raise ValueError ("The ax_or_figsize must be a subplot or a" + 
                              " tuple containing the figure dimensions.")
            
#-----------------------------------------------------------------------------#
# In this section, scanpy library was used.

    def _main_plot(self, main_ax):
        
        """
        Draws circles and colors for Dot Plot
        
        """
        nc = self._tot_iter
        labels = self.labels
        unq_labs = self._mark_keys
        
        sel_mark_ind = self._sel_list_markers.copy()
        iters = [len(sel_mark_ind[ky]) for ky in range(len(sel_mark_ind))]
        min_mark = min(iters)
        self._min_mark = min_mark
                
        #- Arranges elements from the same set one after the other
        srt_clst = []
        for j in range(nc):
            srt_clst.append(np.where(labels == unq_labs[j])[0])
            
        srt_clst = np.hstack(srt_clst)
        
        #- Sort labels for dot plot
        srt_labs = labels[srt_clst]
        cluster_markers = np.concatenate(
            np.array(self._markers.iloc[:,:min_mark])
        )  
         
        logdata = self.data[:, cluster_markers].copy()
        if self._sprs_form:
            logdata.data = np.log2(logdata.data + 1, dtype='f4')
        else:
            logdata = np.log2(logdata + 1, dtype='f4')
            
        #- Data sorted by labels
        logdata = logdata[srt_clst, :]
        #- In-cluster expression rate matrix
        exp_mat = np.zeros((nc, logdata.shape[1]), dtype='f4')
        #- In-cluster mean matrix
        color_mat = np.zeros((nc, logdata.shape[1]), dtype='f4')

        #- Calculation of within-cluster mean and 
        # expression ratios for selected genes
        if self._sprs_form:
            for j in range(nc):
                clst_ind = [srt_labs == unq_labs[j]][0]
                temp = logdata[clst_ind, :]
                exp_mat[j, :] = ((temp != 0).sum(axis=0)).A[0] 
                exp_mat[j, :] = exp_mat[j, :] / temp.shape[0]
                color_mat[j, :] = np.mean(temp, axis=0).A[0] 
        else:
            for j in range(nc):
                clst_ind = [srt_labs == unq_labs[j]][0]
                temp = logdata[clst_ind, :]
                exp_mat[j, :] = ((temp != 0).sum(axis=0))
                exp_mat[j, :] = exp_mat[j, :] / temp.shape[0]
                color_mat[j, :] = np.mean(temp, axis=0)

        scaler = MinMaxScaler()
        rate_sums = np.sum(exp_mat, axis = 0)
        
        #- The highest expression rate corresponds to the largest ring.
        exp_mat = scaler.fit_transform(exp_mat).T     
        sc_rate_sums = np.sum(exp_mat, axis = 1)
        zero_exp = np.where(sc_rate_sums == 0)[0]
        if (len(zero_exp) > 0):
            for i in zero_exp:
                if rate_sums[i] > 0:
                    exp_mat[i,:] = 1
                    
        #- The highest in-cluster mean corresponds to the darkest color
        color_mat = scaler.fit_transform(color_mat).T 
        if self._mean_log:
            color_mat = np.log2(color_mat + 1)
            
        #- Create base color plot
        main_ax.pcolor(color_mat, cmap=self._cmap)
        
        #- For size legend 
        self._dot_max = np.max(exp_mat)
        self._dot_min = np.min(exp_mat)
        exp_mat = exp_mat ** self._size_exp
        exp_mat *= (self._lrg_dot - self._sml_dot) + self._sml_dot
        
        y,x = np.indices(exp_mat.shape)
        y = y.flatten() + 0.5
        x = x.flatten() + 0.5
        
        #- Add Circles
        main_ax.scatter(
            x=x, y=y, 
            s=exp_mat, 
            linewidth=self._dot_edge_lw,
            facecolor='none',
            edgecolor=self._dot_edge_color,
            alpha=0.8,
            norm=Normalize(vmin=None, vmax=None),
            marker='o'
        )
        for axis in ['top', 'bottom', 'left', 'right']:
            main_ax.spines[axis].set_linewidth(1.5)
        y_ticks = np.arange(exp_mat.shape[0]) + 0.5
        main_ax.set_yticks(y_ticks)
        main_ax.set_yticklabels(
            [self.gene_ann[idx] for idx in cluster_markers],
            minor=False
        )
        x_ticks = np.arange(exp_mat.shape[1]) + 0.5
        main_ax.set_xticks(x_ticks)
        main_ax.set_xticklabels(
            unq_labs,
            rotation=90,
            ha='center',
            minor=False,
        )
        main_ax.tick_params(axis='both', labelsize='medium')
        main_ax.grid(self._plt_grid)
        main_ax.set_ylim(exp_mat.shape[0], 0)
        main_ax.set_xlim(0, exp_mat.shape[1])
        main_ax.yaxis.set_tick_params(which='minor', left=False, right=False)
        main_ax.xaxis.set_tick_params(which='minor', top=False, bottom=False)
        main_ax.set_zorder(100)
        
        return main_ax
#-----------------------------------------------------------------------------#
# In this section, scanpy library was used.

    def _plot_bracket(
            self, 
            gene_groups_ax,
            heat_brac: bool = False
    ):
        """
        Draws brackets for dot plot and heatmap
        
        """
        nc = self._tot_iter
        unq_labels = list(self._mark_keys)
        nof_marker = self._min_mark
        
        start = 0
        group_labels = unq_labels
        group_positions = []
        
        if heat_brac:
            
            sng_len = 1/nc
            adj_mar = sng_len/nof_marker
            mov = sng_len - (adj_mar)
            top_adj = -adj_mar/2
            bot_adj = -adj_mar/2
            for pos in range(nc):
                group_positions.append([start, start + mov])
                start = start + sng_len
            txt_args = {'ha':'center', 'va':'top',
                        'rotation': 90, 'fontsize': 'x-large'}
            
        else:
            top_adj = self._top_adj
            bot_adj = self._bot_adj
            mov = nof_marker
            for pos in range(nc):
                group_positions.append([start, start + mov])
                start = start + nof_marker
            txt_args = {'ha':'left', 'va':'center',
                        'rotation': 270, 'fontsize': 'small'}
            
        # get the 'brackets' coordinates as lists of start and end positions
        top = [x[0] - top_adj for x in group_positions]
        bottom = [x[1] - bot_adj for x in group_positions]
        
        # verts and codes are used by PathPatch to make the brackets
        verts = []
        codes = []
        
        for idx, (top_coor, bottom_coor) in enumerate(zip(top, bottom)):
            
            diff = bottom[idx] - top[idx]
            group_y_center = top[idx] + float(diff) / 2
                    
            if heat_brac:
                verts.append((top_coor, 1))  # upper-right
                verts.append((top_coor, 0.5))  # upper-left
                verts.append((bottom_coor,0.5))  # lower-left
                verts.append((bottom_coor,1))  # lower-right
                tx_x = group_y_center
                tx_y = 0.3
            else:
                verts.append((0, top_coor))  # upper-left
                verts.append((0.5,top_coor))  # upper-right
                verts.append((0.5,bottom_coor))  # lower-right
                verts.append((0,bottom_coor))  # lower-left
                tx_x = 0.8
                tx_y = group_y_center
                if hasattr(group_labels[idx], '__len__'):
                    if diff * 2 < len(group_labels[idx]):
                        # cut label to fit available space
                        group_labels[idx] = (
                            group_labels[idx][: int(diff * 2)] +".")
           
            codes.append(Path.MOVETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)
            codes.append(Path.LINETO)
            gene_groups_ax.text(
                x = tx_x,
                y = tx_y,
                s = group_labels[idx],
                **txt_args
            )
            
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        gene_groups_ax.add_patch(patch)
        gene_groups_ax.grid(False)
        gene_groups_ax.axis('off')
        
        # - Remove y ticks
        gene_groups_ax.tick_params(axis='y', left=False, labelleft=False)
        #- Remove x ticks and labels
        gene_groups_ax.tick_params(
            axis='x', 
            bottom=False, 
            labelbottom=False, 
            labeltop=False
        )
        
        return gene_groups_ax
    
#-----------------------------------------------------------------------------#
# In this section, scanpy library was used.

    def _plot_legend(self, legend_ax):
        
        """
        Draws dot plot legends

        """
        dot_max = self._dot_max
        dot_min = self._dot_min
        
        diff = dot_max - dot_min
        
        if 0.3 < diff <= 0.6:
            step = 0.1
        elif diff <= 0.3:
            step = 0.05
        else:
            step = 0.2
            
        # a descending range that is afterwards inverted is used
        # to guarantee that dot_max is in the legend.
        size_range = np.arange(dot_max, dot_min, step * -1)[::-1]
        if dot_min != 0 or dot_max != 1:
            dot_range = dot_max - dot_min
            size_values = (size_range - dot_min) / dot_range
        else:
            size_values = size_range
        
        size = size_values ** self._size_exp
        size = size * (self._lrg_dot - self._sml_dot) + self._sml_dot
        leg_height = self._figsize[1] * 0.1
        cb_height = self._figsize[1] * 0.06
        sp_height = self._figsize[1] * 0.4
        
        height_ratios = [
            self._figsize[1]- leg_height - cb_height - sp_height,
            leg_height,
            sp_height,
            cb_height,
        ]
        
        #- Create legends grid
        fig, legend_gs = self._make_grid_spec(
            legend_ax, nrows=4, ncols=1, height_ratios=height_ratios
        )
        size_legend_ax = fig.add_subplot(legend_gs[1])
        color_legend_ax = fig.add_subplot(legend_gs[3])
        
        #- Add Colorbar
        c_norm = colors.Normalize(vmin=None, vmax=None)
        matplotlib.colorbar.ColorbarBase(
            color_legend_ax, 
            orientation='horizontal', 
            cmap=self._cmap, 
            norm = c_norm
        )
        color_legend_ax.set_title(self._col_leg_title, fontsize='small')
        color_legend_ax.xaxis.set_tick_params(labelsize='small')
        
        #- Add Size legend
        size_legend_ax.scatter(
            x=np.arange(len(size)) + 0.5,
            y=np.repeat(0, len(size)),
            s=size,
            color='gray',
            edgecolor=self._dot_edge_color,
            linewidth=self._dot_edge_lw,
            zorder=100
        )
        
        size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
        labels = ["{}".format(np.round(
            (x * 100),decimals=0).astype(int)) for x in size_range]
        size_legend_ax.set_xticklabels(labels, fontsize='small')
        
        # remove y ticks and labels
        size_legend_ax.tick_params(
            axis='y', left=False, labelleft=False, labelright=False
        )
        
        # remove subplot spines
        for axis in ['top', 'bottom', 'left', 'right']:
            size_legend_ax.spines[axis].set_visible(False)
        size_legend_ax.grid(False)
        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_ylim(-1.05 - self._lrg_dot * 0.003, 4)
        size_legend_ax.set_title(
            self._size_leg_title, 
            y=ymax + 0.45, 
            size='small'
        )
        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)
        
        return legend_ax
#-----------------------------------------------------------------------------#
# In this section, scanpy library was used.

    def dot_plot(
        self, 
        cmap: Optional[str] = _DEFAULT_CMAP,
        figsize: Optional[Tuple[int, int]] = _DEFAULT_FIG_SIZE,
        log_transform: bool = True,
        grid: bool = False,
        plot_legends: bool = True,
        plot_brackets: bool = True,
        size_leg_title: Optional['str'] = _DEFAULT_SIZE_LEGEND_TITLE,
        col_leg_title: Optional['str'] = _DEFAULT_COLOR_LEGEND_TITLE,
        wspace: Optional[float] = _DEFAULT_WSPACE,
        largest_dot: Optional[float] = _DEFAULT_LARGEST_DOT,
        smallest_dot: Optional[float] = _DEFAULT_SMALLEST_DOT,
        size_exponent: Optional[float] = _DEFAULT_SIZE_EXPONENT,
        dot_edge_lw: Optional[float] = _DEFAULT_DOT_EDGELW,
        dot_edge_color: Optional['str'] = _DEFAULT_DOT_EDGECOLOR,
        legends_width: Optional[float] = _DEFAULT_LEGENDS_WIDTH,
        bracket_width: Optional[float] = _DEFAULT_BRACKET_WIDTH,
        bracket_top_adj: Optional[float] = _DEFAULT_TOP_ADJUSTMENT,
        bracket_bot_adj: Optional[float] = _DEFAULT_BOT_ADJUSTMENT,
        plot_name: Optional[str] = _DEFAULT_DOT_NAME,
        save_plot: bool = False,
        plot_dpi: int = 300
    ):
        """\
        This function draws the dot plot from the scanpy library
        In this plot,the colors represent the normalized within-cluster 
        average expression, and the circles represent the within-cluster 
        expression rates.
        
        Parameters
        ----------
        cmap 
            Colormap for dotplot. 
        figsize 
            Figure size. 
        log_transform 
            Performs log normalization before finding in-cluster mean values. 
        grid 
            Draws grid. 
        plot_legends 
            Adds legends. 
        plot_legends 
            Adds brackets. 
        size_leg_title
            Size legend title.
        col_leg_title
            Colorbar title. 
        wspace
            The space between the legends and the main plot. 
        largest_dot
            The size of the largest ring. 
        smallest_dot
            The size of the smallest ring. 
        size_exponent
            The exponential coefficient used for scaling expression rates.
        dot_edge_lw 
            Thickness of the rings. 
        dot_edge_color
            Color of the rings. 
        legends_width
            Width of legends by figure. 
        bracket_width
            Width of legends by figure.
        bracket_top_adj
            Adjustment that allows the brackets to slide up. 
        bracket_bot_adj
            Adjustment that allows the brackets to slide down
        plot_name
            Name of plot to save
        save_plot
            If True, the plot is saved in the working directory.
        plot_dpi
            The dpi value of the figure to be saved
            
        Examples
        --------
        
        .. plot::
           :include-source: True

           >>> import scmags as mg
           >>> li = mg.datasets.li()
           >>> li.filter_genes()
           >>> li.sel_clust_marker()
           >>> li.dot_plot(figsize = (6,12), largest_dot = 200)

        """
        self._plt_grid = grid
        self._mean_log  = log_transform
        
        if cmap != self._cmap or isinstance(cmap, str):
            self._cmap = cm.get_cmap(cmap)
        if figsize != self._figsize:
            self._figsize = figsize
        if size_leg_title != self._size_leg_title:
            self._size_leg_title = size_leg_title
        if col_leg_title != self._col_leg_title:
            self._col_leg_title = col_leg_title
        if wspace != self._wspace:
            self._wspace = wspace
        if smallest_dot != self._sml_dot:
            self._sml_dot = smallest_dot
        if largest_dot != self._lrg_dot:
            self._lrg_dot = largest_dot
        if size_exponent != self._size_exp:
            self._size_exp = size_exponent
        if dot_edge_color != self._dot_edge_color:
            self._dot_edge_color = dot_edge_color
        if dot_edge_lw != self._dot_edge_lw:
            self._dot_edge_lw = dot_edge_lw
        if legends_width != self._legends_width:
            self._legends_width = legends_width
        if bracket_width != self._bracket_width:
            self._bracket_width = bracket_width
        if bracket_top_adj != self._top_adj:
            self._top_adj = bracket_top_adj
        if bracket_bot_adj != self._bot_adj:
            self._bot_adj = bracket_bot_adj
        
        # The space between the legends and the main plot
        legends_width_spacer  = 0.8 / figsize[0]
        mainplot_width = figsize[0] - (self._legends_width)
        fig, gs = self._make_grid_spec(
            ax_or_figsize=figsize,
            nrows=2,
            ncols=2,
            wspace=legends_width_spacer,
            width_ratios=[mainplot_width , self._legends_width],
            height_ratios=[0.98, 0.02]
        )
        if plot_brackets:
            var_groups_height = self._bracket_width
        else: 
            var_groups_height = 0
            
        width_ratios = [mainplot_width, var_groups_height]
        mainplot_gs = GridSpecFromSubplotSpec(
            nrows=1,
            ncols=2,
            wspace=self._wspace,
            hspace=0.0,
            subplot_spec=gs[0, 0],
            width_ratios=width_ratios,
        )
        main_ax = fig.add_subplot(mainplot_gs[0, 0])
        main_ax = self._main_plot(main_ax)
        
        if plot_brackets :
            gene_groups_ax = fig.add_subplot(mainplot_gs[0, 1], sharey=main_ax)
            gene_groups_ax = self._plot_bracket(gene_groups_ax)
        if plot_legends:
            legend_ax = fig.add_subplot(gs[0, 1])
            legend_ax = self._plot_legend(legend_ax)
            
        plt.show()
        if save_plot:
            plot_path = os.getcwd() + '/' + plot_name + '.svg'
            fig.savefig(plot_path, dpi=plot_dpi)
#-----------------------------------------------------------------------------#