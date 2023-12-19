import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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


def get_radar_coords(lon, lat, radar):
    return pyart.core.geographic_to_cartesian_aeqd(
        lon, lat, radar.longitude["data"].item(), radar.latitude["data"].item()
    )


def plot_ppi_fig(
    radar,
    qtys,
    ncols=2,
    max_dist=100,
    range_rings_sep=25,
    outdir=Path("."),
    ext="png",
    markers=None,
    title_prefix="",
    extent=None,
):
    """Plot a figure of 4 radar variables.

    Parameters
    ----------
    radar : pyart.core.Radar
        The radar object.
    qtys : list of str
        List of radar variables to plot.
    ncols : int, optional
        Number of columns, by default 2
    max_dist : int, optional
        Maximum distance to plot, by default 100
    range_rings_sep : int, optional
        Distance between range rings, by default 25
    outdir : pathlib.Path , optional
        Output path, by default Path(".")
    ext : str, optional
        Figure filename extension, by default "png"
    markers : list, optional
        List of marker tuples as (lon, lat, marker_color, marker_symbol), by default None
    title_prefix : str, optional
        Prefix to add to the figure title, by default ""
    extent : list, optional
        Extent of the plot in km from radar location, as xmin xmax ymin ymax, by default None

    """
    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
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
        figsize=(6 * ncols, 5 * nrows),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    display = pyart.graph.RadarDisplay(radar)
    time = datetime.strptime(radar.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ")
    fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")

    if markers is not None:
        markers = [
            (*get_radar_coords(float(m[0]), float(m[1]), radar), m[2], m[3])
            for m in markers
        ]

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

            # Set 0-values as nan to prevent them from being
            # plotted with same color as 1
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

        if markers is not None:
            for marker in markers:
                ax.plot(
                    marker[0] / 1000,
                    marker[1] / 1000,
                    marker=marker[3],
                    markerfacecolor="none",
                    markeredgecolor=marker[2],
                    zorder=20,
                )

        for r in np.arange(range_rings_sep, 500, range_rings_sep):
            display.plot_range_ring(r, ax=ax, lw=0.5, col="k")
        ax.set_title(utils.TITLES[qty])

    for ax in axes.flat:
        # x and y labels
        ax.set_ylabel("Distance from radar (km)")
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel("Distance from radar (km)")
        ax.xaxis.set_major_formatter(fmt)
        ax.label_outer()

        # Set limits
        if extent is not None:
            ax.set_xlim([extent[0], extent[1]])
            ax.set_ylim([extent[2], extent[3]])
        else:
            ax.set_xlim([-max_dist, max_dist])
            ax.set_ylim([-max_dist, max_dist])
        ax.set_aspect(1)
        ax.grid(zorder=15, linestyle="-", linewidth=0.4)
        ax.patch.set_facecolor("white")

    # Remove empty axes
    for ax in axes.flat[n_qtys:]:
        ax.remove()

    fig.suptitle(
        f"{title_prefix}{time:%Y/%m/%d %H:%M} "
        f"UTC {radar.fixed_angle['data'][0]:.1f}°",
        y=1.015,
    )

    fname = outdir / (
        f"radar_vars_{radar.metadata['instrument_name'].decode().lower()[:3]}_"
        f"{time:%Y%m%d%H%M%S}_{radar.fixed_angle['data'][0]:.1f}.{ext}"
    )

    fig.savefig(fname, bbox_inches="tight")


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
        help="Quantities to be plotted, in ODIM short names.",
    )
    argparser.add_argument(
        "--markers",
        type=str,
        nargs="+",
        default=[],
        help="marker locations as lon,lat,color,marker",
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
        "--extent", type=float, nargs=4, help="Extent of the plot in km from radar location, as xmin xmax ymin ymax"
    )
    argparser.add_argument(
        "--ext",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Output image extension",
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

    radar = pyart.io.read(infile)

    markers = None
    if len(args.markers):
        markers = [m.split(",") for m in args.markers]

    plot_ppi_fig(
        radar,
        args.qtys,
        ncols=args.ncols,
        max_dist=args.rmax,
        range_rings_sep=args.range_rings,
        outdir=outdir,
        markers=markers,
        ext=args.ext,
        title_prefix=f"{radar.metadata['instrument_name'].decode().capitalize()} ",
        extent=args.extent,
    )
