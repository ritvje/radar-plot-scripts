import logging
import argparse
import pyart
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime
from pathlib import Path

import utils

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def plot_rhi_fig(
    radar,
    qtys,
    ncols=2,
    max_dist=100,
    range_rings_sep=25,
    outdir=Path("."),
    ext="png",
    hmax=10,
):
    """Plot a RHI figure.

    Parameters
    ----------
    radar : pyart.core.Radar
        Radar object.
    qtys : list
        Quantities to plot.
    ncols : int
        Number of columns in figure.
    max_dist : int
        Maximum distance to plot in km.
    range_rings_sep : int
        Distance between range rings in km.
    outdir : pathlib.Path
        Output directory.
    ext : str
        Output image extension.
    hmax : int
        Maximum height for images in km.

    """
    cbar_ax_kws = {
        "width": "3%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }

    n_qtys = len(qtys)
    nrows = np.ceil(n_qtys / ncols).astype(int)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 7, 5 * nrows),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    display = pyart.graph.RadarDisplay(radar)
    time = datetime.strptime(radar.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ")
    fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")

    for ax, qty in zip(axes.flat, qtys):
        cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

        if "VRAD" in qty:
            utils.QTY_RANGES[qty] = (
                -1 * np.ceil(radar.get_nyquist_vel(0)),
                np.ceil(radar.get_nyquist_vel(0)),
            )

        cmap, norm = utils.get_colormap(qty)
        cbar_ticks = None

        if norm is None:
            # define the bins and normalize
            bounds = np.linspace(utils.QTY_RANGES[qty][0], utils.QTY_RANGES[qty][1], 40)
            norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
            cmap = plt.get_cmap(cmap, len(bounds))
        elif isinstance(norm, mpl.colors.BoundaryNorm):
            cbar_ticks = norm.boundaries

        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            format=mpl.ticker.StrMethodFormatter(utils.QTY_FORMATS[qty]),
            orientation="vertical",
            cax=cax,
            ax=None,
            ticks=cbar_ticks,
        )
        cbar.set_label(label=utils.COLORBAR_TITLES[qty])

        if qty == "HCLASS":
            utils.set_HCLASS_cbar(cbar)

            # Set 0-values as nan to prevent them from
            # being plotted with same color as 1
            radar.fields["radar_echo_classification"]["data"].set_fill_value(np.nan)

            radar.fields["radar_echo_classification"]["data"] = np.ma.masked_values(
                radar.fields["radar_echo_classification"]["data"], 0
            )

        display.plot(
            utils.PYART_FIELDS[qty],
            0,
            title="",
            ax=ax,
            axislabels_flag=False,
            colorbar_flag=False,
            cmap=cmap,
            norm=norm,
            zorder=10,
        )

        for r in np.arange(range_rings_sep, max_dist, range_rings_sep):
            display.plot_range_ring(r, ax=ax, lw=0.5, col="k")
        ax.set_title(utils.TITLES[qty])

    for ax in axes.flat:
        ax.set_xlabel("Distance from radar (km)")
        ax.xaxis.set_major_formatter(fmt)

        ax.set_ylabel("Altitude from radar (km)")
        ax.yaxis.set_major_formatter(fmt)

        ax.label_outer()

        ax.set_ylim([0, hmax])
        ax.set_xlim([0, max_dist])
        ax.grid(zorder=15, linestyle="-", linewidth=0.3)

    fig.suptitle(
        f"{time:%Y/%m/%d %H:%M} UTC {radar.fixed_angle['data'][0]:.1f}°", y=0.92
    )

    fname = outdir / (
        f"radar_rhi_{radar.metadata['instrument_name'].decode().lower()[:3]}_"
        f"{time:%Y%m%d%H%M%S}_{radar.fixed_angle['data'][0]:.1f}.{ext}"
    )

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(fname)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("inpath", type=str, help="Path to file")
    argparser.add_argument("--out", type=str, help="Output path", default=".")
    argparser.add_argument(
        "--qtys",
        type=str,
        nargs="+",
        default=[
            "DBZH",
            "VRAD",
            "RHOHV",
            "ZDR",
        ],
        help="Quantities (4) to be plotted, in ODIM short names.",
    )
    argparser.add_argument(
        "--rmax", type=int, default=100, help="Maximum range in kilometers"
    )
    argparser.add_argument(
        "--range-rings", type=int, default=25, help="Range rings separation in km"
    )
    argparser.add_argument(
        "--ncols", type=int, default=2, help="Number of columns in the figure"
    )
    argparser.add_argument(
        "--ext",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Output image extension",
    )
    argparser.add_argument(
        "--hmax", type=int, default=10, help="Maximum height for images in km"
    )
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Get style from environment variable
    style = os.environ.get("MPLSTYLE")
    if style is not None:
        plt.style.use(style)
    pyart.load_config(os.environ.get("PYART_CONFIG"))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    infile = Path(args.inpath).resolve()

    radar = pyart.io.read_sigmet(infile)

    plot_rhi_fig(
        radar,
        args.qtys,
        ncols=args.ncols,
        max_dist=args.rmax,
        range_rings_sep=args.range_rings,
        outdir=outdir,
        hmax=args.hmax,
    )
