import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def combine_plots(planes, **kwargs):
    """
    Combines multiple wind plots in one figure
    Dimensions of "planes": [[sim1_plane1, ..], [sim2,plane1, ..]]
    kwargs:
        time            default: max(plane.time)
        fignum          default: 1
        vmin            default: None
        vmax            default: None
        xlim            default: None
        ylim            default: None
        zlim            default: None
        hub_height      default: 240/150
        rot_diam        default: 1
        turb_loc        default: [0, 0]
        colormap        default: viridis
    """
    def check(attribute,default=None):
        if not attribute in kwargs.keys():
            kwargs[attribute] = default
        return kwargs[attribute]

    figs = {}
    axes = {}
    if not isinstance(planes,list): planes = [[planes]]
    if not isinstance(planes[0],list): planes = [planes]

    Nplanes = len(planes[0])
    Nsims = len(planes)

    for k in range(Nsims):
        figs[kwargs['fignum']+k] = plt.figure(k+check('fignum',1),figsize=(9.5,4))
        for n in range(Nplanes):
            z_N = planes[k][n].z_N
            for i in range(z_N):
                # Plot flow field
                plt.set_cmap(check('colormap','viridis'))
                axes[n,k] = plt.subplot(Nplanes,z_N,n*z_N+i+1)
                if not check('mean'): 
                    planes[k][n].plot_plane(planes[k][n].z[i],check('time',np.max(planes[k][n].time)),\
                                                ax=axes[n,k],vmin=check('vmin'),vmax=check('vmax'))
                else:
                    planes[k][n].plot_mean_plane(planes[k][n].z[i],ax=axes[n,k],vmin=check('vmin'),vmax=check('vmax'))
                # Remove subplot colorbar
                im = [obj for obj in axes[n,k].get_children() if isinstance(obj, mpl.collections.Collection)][0]
                im.colorbar.remove()
                # Set axis limits
                axes[n,k].set_xlim(check('xyz'[np.argmax(planes[k][n].x_dir)]+'lim',\
                                         [np.min(planes[k][n].x),np.max(planes[k][n].x)]))
                axes[n,k].set_ylim(check('xyz'[np.argmax(planes[k][n].y_dir)]+'lim',\
                                         [np.min(planes[k][n].y),np.max(planes[k][n].y)]))
                # Remove unnecessary labels
                if i>0:
                    axes[n,k].set_ylabel('')
                # Plot turbine - change later to possibly add multiple turbines
                if planes[k][n].z_dir[2] == 0: planedir = 'yz'
                else: 
                    planedir = 'xy'
                    axes[n,0].plot([np.min(planes[k][n].x),np.max(planes[k][n].x)],\
                                   [check('turb_loc',[0, 0])[1],kwargs['turb_loc'][1]],'w-.',linewidth=0.25)
                planes[k][n].plot_turbine(check('hub_height',150/240),check('rot_diam',1),\
                                          kwargs['turb_loc'],ax=axes[n,k],plane=planedir)
                
        figs[kwargs['fignum']+k].tight_layout()

        # Set colorbar
        figs[kwargs['fignum']+k].subplots_adjust(right=0.88)
        cbar_ax = figs[kwargs['fignum']+k].add_axes([0.92, 0.23, 0.02, 0.55])
        cb = figs[kwargs['fignum']+k].colorbar(im,cax=cbar_ax,shrink=0.6)