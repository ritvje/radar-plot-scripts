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
import h5py

import xradar

import utils


def get_radar_coords(lon, lat, radar):
    return pyart.core.geographic_to_cartesian_aeqd(
        lon, lat, radar.longitude["data"].item(), radar.latitude["data"].item()
    )


def plot_ppi_fig(
    radar,
    gatefilter,
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
    nrows = n_qtys

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows),
        sharex="col",
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )
    display = pyart.graph.RadarDisplay(radar)
    time = datetime.strptime(radar.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ")
    fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")
    props = dict(boxstyle="square", facecolor="white", alpha=0.9)

    if markers is not None:
        markers = [(*get_radar_coords(float(m[0]), float(m[1]), radar), *m[2:]) for m in markers]

        # remove markers that are beyond the radar range
        markers = [m for m in markers if np.linalg.norm(m[:2]) < max_dist * 1000]

    for i, qty in enumerate(qtys):
        ax = axes[i, 0]
        ax_filter = axes[i, 1]
        cax = inset_axes(ax_filter, bbox_transform=ax_filter.transAxes, **cbar_ax_kws)

        if "VRAD" in qty:
            # utils.QTY_RANGES[qty] = (
            #     -1 * np.ceil(radar.get_nyquist_vel(0)),
            #     np.ceil(radar.get_nyquist_vel(0)),
            # )
            # Cheap solution for now since nyquist velocity is not available in radar object from h5 file
            if np.nanmax(radar.fields[utils.PYART_FIELDS_ODIM[qty]]["data"]) > 8:
                vmax = 48
            else:
                vmax = 8
            utils.QTY_RANGES[qty] = (-vmax, vmax)

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

        for ax_, filt, title in zip(axes[i], [None, gatefilter], ["Alkuperäinen", "Suodatettu"]):

            display.plot(
                utils.PYART_FIELDS_ODIM[qty],
                0,
                title="",
                ax=ax_,
                axislabels_flag=False,
                colorbar_flag=False,
                cmap=cmap,
                norm=norm,
                zorder=10,
                gatefilter=filt,
                raster=True,
            )

            if markers is not None:
                for marker in markers:
                    ax_.plot(
                        marker[0] / 1000,
                        marker[1] / 1000,
                        linestyle="None",
                        marker=marker[3],
                        markerfacecolor="none",
                        markeredgecolor=marker[2],
                        markersize=marker[4],
                        zorder=20,
                        alpha=marker[5],
                        markeredgewidth=marker[6],
                        markerfacecoloralt="none",
                        fillstyle="none",
                    )

            for r in np.arange(range_rings_sep, 500, range_rings_sep):
                display.plot_range_ring(r, ax=ax_, lw=0.5, col="k")
            # ax_.set_title(utils.TITLES[qty])
            ax_.text(
                0.05,
                0.95,
                title,
                transform=ax_.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
                zorder=100,
            )

    for ax in axes.flat:
        # x and y labels
        # ax.set_ylabel("Distance from radar (km)")
        ax.yaxis.set_major_formatter(fmt)
        # ax.set_xlabel("Distance from radar (km)")
        ax.xaxis.set_major_formatter(fmt)
        # plt.setp(ax.get_xticklabels()[-1], visible=False)
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

    for ax in axes[:, -1]:
        # Remove y tick lines
        ax.yaxis.set_ticks_position("none")

    # # Remove empty axes
    # for ax in axes.flat[n_qtys:]:
    #     ax.remove()

    fig.subplots_adjust(wspace=0.0, hspace=0.05)
    fig.suptitle(
        f"{title_prefix}{time:%Y/%m/%d %H:%M} " f"UTC {radar.fixed_angle['data'].item():.1f}°",
        y=0.92,
    )

    fname = outdir / (
        f"radar_vars_{title_prefix.strip().lower()}_"
        f"{time:%Y%m%d%H%M%S}_{radar.fixed_angle['data'].item():.1f}.{ext}"
    )

    fig.savefig(fname, bbox_inches="tight")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("inpath", type=str, help="Path to file")
    argparser.add_argument("maskfile", type=str, help="Path to mask file")
    argparser.add_argument("-d", "--datasets", type=int, nargs="+", default=[1], help="Dataset numbers to plot")
    argparser.add_argument("--out", type=str, help="Output path", default=".")
    argparser.add_argument(
        "--qtys",
        type=str,
        nargs="+",
        default=[
            "DBZH",
            "VRADH",
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
        help="marker locations as lon,lat,color,marker,size",
    )
    argparser.add_argument("--markerlist", type=str, help="csv file with marker information as lon,lat")
    argparser.add_argument("--marker-color", type=str, default="black", help="Marker color")
    argparser.add_argument("--marker-symbol", type=str, default="o", help="Marker symbol")
    argparser.add_argument("--marker-size", type=int, default=2, help="Marker size")
    argparser.add_argument("--marker-alpha", type=float, default=1, help="Marker alpha")
    argparser.add_argument("--marker-edgewidth", type=float, default=1, help="Marker edge width")
    argparser.add_argument("--rmax", type=int, default=100, help="Maximum range in kilometers")
    argparser.add_argument("--range-rings", type=int, default=25, help="Range rings separation in km")
    argparser.add_argument("--ncols", type=int, default=2, help="Number of columns in the figure")
    argparser.add_argument(
        "--extent", type=float, nargs=4, help="Extent of the plot in km from radar location, as xmin xmax ymin ymax"
    )
    argparser.add_argument(
        "--ext",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
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
    mask_file = Path(args.maskfile).resolve()

    markers = None
    if len(args.markers):
        markers = [m.split(",") for m in args.markers]

    # Read markers from file
    if args.markerlist is not None:
        markers = markers or []
        with open(args.markerlist, "r") as f:
            for line in f:
                markers.append(
                    [
                        *[float(f) for f in line.strip().split(",")],
                        args.marker_color,
                        args.marker_symbol,
                        args.marker_size,
                        args.marker_alpha,
                        args.marker_edgewidth,
                    ]
                )

    for dataset in args.datasets:

        radar = pyart.aux_io.read_odim_h5(infile, include_datasets=[f"dataset{dataset}"])

        # read mask array
        with h5py.File(mask_file, "r") as f:
            quality = "quality2"
            mask_ = f[f"dataset{dataset}/{quality}/data"][:]
            offset = f[f"dataset{dataset}/{quality}/what"].attrs["offset"]
            gain = f[f"dataset{dataset}/{quality}/what"].attrs["gain"]
            nodata = f[f"dataset{dataset}/{quality}/what"].attrs["nodata"]
            undetect = f[f"dataset{dataset}/{quality}/what"].attrs["undetect"]

            # unpack
            mask = mask_ * gain + offset
            mask[mask_ == nodata] = np.nan
            mask[mask_ == undetect] = np.nan

        gatefilter = pyart.filters.GateFilter(radar)
        gatefilter.exclude_gates(mask == 1)

        title_prefix = radar.metadata["source"].split(",")[-1].split(":")[-1]

        plot_ppi_fig(
            radar,
            gatefilter,
            args.qtys,
            ncols=args.ncols,
            max_dist=args.rmax,
            range_rings_sep=args.range_rings,
            outdir=outdir,
            markers=markers,
            ext=args.ext,
            title_prefix=f"{title_prefix} ",
            extent=args.extent,
        )
