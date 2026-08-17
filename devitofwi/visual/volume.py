__all__ = ["plot_slices"]

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_slices(
    vp, x, y, z,
    ix=None, iy=None, iz=None,
    vmin=None, vmax=None,
    cmap='jet',
    title=None,
    cbar_label='Vp [km/s]',
    scale=1.2,
    savepath=None,
    dpi=150,
):
    """Plot three orthogonal slices of a 3D volume in a corner layout.

    The layout shows three slices through a common ``(ix, iy, iz)`` voxel:
    a depth-slice (x-y plane) at the top-left, an inline section (x-z plane)
    at the bottom-left, and a crossline section (y-z plane) at the bottom-right.
    Dashed lines on each panel indicate the positions of the other two slices.

    Parameters
    ----------
    vp : :obj:`numpy.ndarray`
        3D volume of shape ``(nx, ny, nz)``
    x : :obj:`numpy.ndarray`
        Inline axis of length ``nx`` in km
    y : :obj:`numpy.ndarray`
        Crossline axis of length ``ny`` in km
    z : :obj:`numpy.ndarray`
        Depth axis of length ``nz`` in km
    ix : :obj:`int`, optional
        Inline slice index (default: center of x-axis)
    iy : :obj:`int`, optional
        Crossline slice index (default: center of y-axis)
    iz : :obj:`int`, optional
        Depth slice index (default: ``nz // 4``)
    vmin : :obj:`float`, optional
        Lower colorbar limit (default: ``vp.min()``)
    vmax : :obj:`float`, optional
        Upper colorbar limit (default: ``vp.max()``)
    cmap : :obj:`str`, optional
        Matplotlib colormap name
    title : :obj:`str`, optional
        Figure title
    cbar_label : :obj:`str`, optional
        Colorbar label
    scale : :obj:`float`, optional
        Figure size multiplier
    savepath : :obj:`str`, optional
        If provided, save the figure to this path
    dpi : :obj:`int`, optional
        Resolution used when saving

    Returns
    -------
    fig : :obj:`matplotlib.figure.Figure`
        The created figure
    axes : :obj:`tuple`
        Tuple ``(ax_map, ax_xz, ax_yz)`` of the three slice axes

    """
    nx, ny, nz = vp.shape
    ix = ix if ix is not None else nx // 2
    iy = iy if iy is not None else ny // 2
    iz = iz if iz is not None else nz // 4
    vmin = vmin if vmin is not None else vp.min()
    vmax = vmax if vmax is not None else vp.max()

    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    Lz = z[-1] - z[0]

    fig_w = (Lx + Ly) * scale + 1.5
    fig_h = (Ly + Lz) * scale

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[Lx, Ly],
        height_ratios=[Ly, Lz],
        hspace=0.04,
        wspace=0.04,
    )

    ax_map = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[1, 0], sharex=ax_map)
    ax_yz = fig.add_subplot(gs[1, 1], sharey=ax_xz)

    imkw = dict(cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')

    ax_map.imshow(
        vp[:, :, iz].T, **imkw,
        extent=[x[0], x[-1], y[0], y[-1]], origin='lower',
    )
    ax_map.axhline(y[iy], color='cyan', lw=1, ls='--')
    ax_map.axvline(x[ix], color='steelblue', lw=1, ls='--')
    ax_map.set_ylabel('Crossline (km)')
    ax_map.tick_params(labelbottom=False)

    ax_xz.imshow(
        vp[:, iy, :].T, **imkw,
        extent=[x[0], x[-1], z[-1], z[0]],
    )
    ax_xz.axvline(x[ix], color='steelblue', lw=1, ls='--')
    ax_xz.set_xlabel('Inline (km)')
    ax_xz.set_ylabel('Depth (km)')

    im = ax_yz.imshow(
        vp[ix, :, :].T, **imkw,
        extent=[y[0], y[-1], z[-1], z[0]],
    )
    ax_yz.axvline(y[iy], color='cyan', lw=1, ls='--')
    ax_yz.set_xlabel('Crossline (km)')
    ax_yz.tick_params(labelleft=False)

    ax_cb.set_axis_off()
    cbar = fig.colorbar(im, ax=ax_cb, fraction=0.5, pad=0.05, shrink=0.85)
    cbar.set_label(cbar_label)
    if title:
        ax_cb.text(0.05, 0.98, title,
                   transform=ax_cb.transAxes,
                   fontsize=11, va='top', ha='left',
                   fontweight='bold')

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')

    return fig, (ax_map, ax_xz, ax_yz)
