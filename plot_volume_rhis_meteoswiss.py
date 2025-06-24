"""Plot radar volumes from RAW data."""

import argparse
import re
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
import dask.bag as db
import pandas as pd
import matplotlib as mpl
from collections import defaultdict
from PIL import Image

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pyart
import struct

import utils

# import plot_utils
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def plot_volume_tasks(
    files,
    azimuths,
    path,
    outpath=".",
    quantities=["DBZH", "VRAD"],
    rmax=100,
    hmax=10,
    dpi=300,
):

    SMALL_SIZE = 4
    BIGGER_SIZE = 8

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Define figure
    ncols = len(quantities)
    nrows = len(azimuths)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3 * ncols, 2.5 * nrows),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }

    min_time = None
    max_time = None

    # Merge radar objects
    full_radar = pyart.aux_io.read_metranet(
        os.path.join(path, files[0]), reader="python", field_names=utils.METRANET_FIELD_NAMES
    )
    max_nyq = full_radar.get_nyquist_vel(0)
    for fn in files[1:]:
        radar = pyart.aux_io.read_metranet(
            os.path.join(path, fn), reader="python", field_names=utils.METRANET_FIELD_NAMES
        )
        full_radar = pyart.util.join_radar(full_radar, radar)
        max_nyq = max(max_nyq, radar.get_nyquist_vel(0))

    display = pyart.graph.RadarDisplay(full_radar)
    for row, azim in enumerate(azimuths):
        dtime = datetime.strptime(radar.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ")
        starttime = dtime + timedelta(seconds=radar.time["data"].min())
        endtime = dtime + timedelta(seconds=radar.time["data"].max())
        if min_time is None or starttime < min_time:
            min_time = starttime
        if max_time is None or endtime > max_time:
            max_time = endtime

        fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")
        for ax, qty in zip(axes[row, :].flat, quantities):
            # Create inset axis for colorbar
            cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

            if "VRAD" in qty:
                utils.QTY_RANGES[qty] = (
                    -1 * np.ceil(max_nyq),
                    np.ceil(max_nyq),
                )

            # Get norm and colormap for the quantity
            cmap, norm = utils.get_colormap(qty)
            cbar_ticks = None

            if norm is None:
                # define the bins and normalize
                bounds = np.linspace(utils.QTY_RANGES[qty][0], utils.QTY_RANGES[qty][1], 40)
                norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
                cmap = plt.get_cmap(cmap, len(bounds))
            elif isinstance(norm, mpl.colors.BoundaryNorm):
                cbar_ticks = norm.boundaries

            # Add colorbar to axis we created before
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

                # Set 0-values as nan to prevent them from being plotted with same color as 1
                radar.fields["radar_echo_classification"]["data"].set_fill_value(np.nan)

                radar.fields["radar_echo_classification"]["data"] = np.ma.masked_values(
                    radar.fields["radar_echo_classification"]["data"], 0
                )

            # Plot the radar image
            try:
                display.plot_azimuth_to_rhi(
                    utils.PYART_FIELDS_MCH[qty],
                    target_azimuth=azim,
                    title="",
                    # vmin=plot_utils.QTY_RANGES[qty][0],
                    # vmax=plot_utils.QTY_RANGES[qty][1],
                    ax=ax,
                    axislabels_flag=False,
                    colorbar_flag=False,
                    cmap=cmap,
                    norm=norm,
                    zorder=10,
                    raster=True,
                )

            except KeyError:
                pass

            # Add some range rings, add more to the list as you wish
            # display.plot_range_rings(
            #     [50, 100, 150, 200, 250], ax=ax, lw=0.2, col="gray")
            # Could also add grid lines
            # display.plot_grid_lines(ax=ax, col="grey", ls=":")
            ax.set_title(
                f"{starttime:%H:%M:%S} - {endtime:%H:%M:%S} " f"{qty} {azim:.1f} deg",
                y=1.01,
                weight="bold",
            )

    # In all axis, set up rax range and aspect
    for ax in axes.flat:
        # ax.set_title(ax.get_title(), y=1.02)
        ax.set_xlim([0, rmax])
        ax.set_ylim([0, hmax])
        # ax.set_aspect(1)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.grid(zorder=0, linestyle="-", linewidth=0.3)

    # title = display.generate_title(field="reflectivity", sweep=0).split("\n")[0]
    fig.suptitle(
        f"{min_time:%Y/%m/%d %H:%M} - {max_time:%Y/%m/%d %H:%M} UTC\n",
        y=0.92,
        weight="bold",
    )
    fig.supxlabel("\nDistance from radar (km)", y=0.08, weight="bold")
    fig.supylabel("Distance from radar (km)", x=0.1, weight="bold")

    # Some adjusting to make plot a bit tighter, adjust as needed
    fig.subplots_adjust(wspace=0.35, hspace=0.2)
    filename = f"rhis_{rmax}_{min_time:%Y%m%d%H%M}.png"
    fig.savefig(
        os.path.join(outpath, filename),
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.clear()
    plt.close(fig)

    # Reduce size by converting colorspace with PIL
    utils.convert_colorspace(os.path.join(outpath, filename))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("inpath", type=str, help="Path to radar directory")
    argparser.add_argument("outpath", type=str, help="Output path")
    argparser.add_argument(
        "azimuths",
        type=float,
        nargs="+",
        help="Azimuths to plot",
    )
    argparser.add_argument("--starttime", type=str, help="Start time for plotting")
    argparser.add_argument("--endtime", type=str, help="End time for plotting")
    argparser.add_argument(
        "--file_regex",
        type=str,
        default="ML([A-Z]{1})([0-9]{9})0U.([0-9]{3})",
        help="Regex for finding volume files",
    )
    argparser.add_argument(
        "--datetime_format",
        type=str,
        default="%y%j%H%M",
        help="Datetime format for extracting time from filename",
    )
    argparser.add_argument("--rmax", type=int, default=100, help="Maximum range for images in km")
    argparser.add_argument("--hmax", type=int, default=10, help="Maximum height for images in km")
    argparser.add_argument("--dpi", type=int, default=500, help="DPI for output images")
    argparser.add_argument(
        "--qtys",
        type=str,
        nargs="+",
        default=[
            "DBZH",
            "VRAD",
            "WRAD",
            "ZDR",
            "PHIDP",
            "RHOHV",
            "SNR",
        ],
        help="Quantities to be plotted, in ODIM short names.",
    )
    argparser.add_argument(
        "--n_workers",
        type=int,
        default=3,
        help="Number of processes to use (if 1, no multi-processing)",
    )
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Get style from environment variable
    style = os.environ.get("MPLSTYLE")
    if style is not None:
        plt.style.use(style)
    pyart.load_config(os.environ.get("PYART_CONFIG"))

    Path(args.outpath).mkdir(parents=True, exist_ok=True)

    if args.starttime is not None:
        args.starttime = datetime.strptime(args.starttime, "%Y%m%d%H%M")
    if args.endtime is not None:
        args.endtime = datetime.strptime(args.endtime, "%Y%m%d%H%M")

    # All sigmet files in inpath
    file_regex = args.file_regex.encode().decode("unicode_escape")
    prog = re.compile(file_regex.encode().decode("unicode_escape"))
    inpath = Path(args.inpath)

    filetime = lambda fn: utils.parse_time_from_filename(fn, file_regex, args.datetime_format, 1)
    volumes = defaultdict(list)
    for file in inpath.glob("*"):
        result = prog.match(file.name)
        if result is None:
            continue

        date = filetime(file.name)
        if args.starttime is not None and date < args.starttime:
            continue
        if args.endtime is not None and date > args.endtime:
            continue
        radar_name = result.group(1)
        volumes[date].append(file)

    # Volumes to list
    volumes = list(volumes.values())

    # Drop empty volumes
    volumes = [sorted(v) for v in volumes if len(v) > 0]

    plot_args = dict(
        azimuths=args.azimuths,
        path=args.inpath,
        outpath=args.outpath,
        quantities=args.qtys,
        rmax=args.rmax,
        hmax=args.hmax,
        dpi=args.dpi,
    )

    if args.n_workers == 1:
        scheduler = "single-threaded"
    else:
        scheduler = "processes"

    # Run all volumes
    bag = db.from_sequence(volumes)
    bag.map(plot_volume_tasks, **plot_args).compute(num_workers=args.n_workers, scheduler=scheduler)
