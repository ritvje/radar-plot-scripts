"""Plot radar volumes from RAW data."""
import argparse
import re
import os
from datetime import datetime, timedelta
import logging
import dask.bag as db
import pandas as pd
from pathlib import Path
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pyart

import utils

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def plot_volume_tasks(
    files, path, outpath=".", quantities=["DBZH", "VRAD"], rmax=100, range_rings_sep=50
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
    nrows = len(files)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3 * ncols, 2.5 * nrows),
        sharex="col",
        sharey="row",
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
    for row, fn in enumerate(sorted(files)):
        radar = pyart.io.read_sigmet(
            os.path.join(path, fn),
        )

        if "SNR" in quantities and "signal_to_noise_ratio" not in radar.fields.keys():
            utils.add_estimated_SNR(os.path.join(path, fn), radar)

        dtime = datetime.strptime(
            radar.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ"
        )
        starttime = dtime + timedelta(seconds=radar.time["data"].min())
        endtime = dtime + timedelta(seconds=radar.time["data"].max())
        if min_time is None or starttime < min_time:
            min_time = starttime
        if max_time is None or endtime > max_time:
            max_time = endtime

        display = pyart.graph.RadarDisplay(radar)
        fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")
        for ax, qty in zip(axes[row, :].flat, quantities):
            # Create inset axis for colorbar
            cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

            if "VRAD" in qty:
                utils.QTY_RANGES[qty] = (
                    -1 * np.ceil(radar.get_nyquist_vel(0)),
                    np.ceil(radar.get_nyquist_vel(0)),
                )

            # Get norm and colormap for the quantity
            cmap, norm = utils.get_colormap(qty)
            cbar_ticks = None

            if norm is None:
                # define the bins and normalize
                bounds = np.linspace(
                    utils.QTY_RANGES[qty][0], utils.QTY_RANGES[qty][1], 40
                )
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

                # Set 0-values as nan to prevent them from being plotted with same color as 1
                radar.fields["radar_echo_classification"]["data"].set_fill_value(np.nan)

                radar.fields["radar_echo_classification"]["data"] = np.ma.masked_values(
                    radar.fields["radar_echo_classification"]["data"], 0
                )

            # Plot the radar image
            display.plot(
                utils.PYART_FIELDS[qty],
                0,
                title="",
                vmin=utils.QTY_RANGES[qty][0],
                vmax=utils.QTY_RANGES[qty][1],
                ax=ax,
                axislabels_flag=False,
                colorbar_flag=False,
                cmap=cmap,
                norm=norm,
                zorder=10,
            )

            # Add some range rings, add more to the list as you wish
            display.plot_range_rings(
                np.arange(range_rings_sep, rmax, range_rings_sep),
                ax=ax,
                lw=0.2,
                col="gray",
            )
            # Could also add grid lines
            ax.set_title(
                f"{qty} {radar.fixed_angle['data'][0]:.1f} deg", y=1.01, weight="bold"
            )
        del display
        del radar

    # In all axis, set up rax range and aspect
    for ax in axes.flat:
        # ax.set_title(ax.get_title(), y=1.02)
        ax.set_xlim([-rmax, rmax])
        ax.set_ylim([-rmax, rmax])
        ax.set_aspect(1)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.grid(zorder=0, linestyle="-", linewidth=0.3)

    # title = display.generate_title(field="reflectivity", sweep=0).split("\n")[0]
    fig.suptitle(
        f"{min_time:%Y/%m/%d %H:%M} - {max_time:%Y/%m/%d %H:%M} UTC",
        y=0.91,
        weight="bold",
    )
    fig.supxlabel("Distance from radar (km)", y=0.08, weight="bold")
    fig.supylabel("Distance from radar (km)", x=0.08, weight="bold")

    # Some adjusting to make plot a bit tighter, adjust as needed
    fig.subplots_adjust(wspace=0, hspace=0.2)

    out = Path(outpath)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"volume_{min_time:%Y%m%d%H%M}.png", dpi=300, bbox_inches="tight")
    fig.clear()
    plt.close(fig)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("inpath", type=str, help="Path to xband directory")
    argparser.add_argument("outpath", type=str, help="Output path")
    argparser.add_argument(
        "--task_regex",
        type=str,
        default="MWS-PPI1_([A-Z]{1})",
        help="Regex for finding volume tasks",
    )
    argparser.add_argument(
        "--file_regex",
        type=str,
        default="WRS([0-9]{12}).RAW([A-Z0-9]{4})",
        help="Regex for finding volume tasks",
    )
    argparser.add_argument(
        "--datetime_format",
        type=str,
        default="%y%m%d%H%M%S",
        help="Datetime format for extracting time from filename",
    )
    argparser.add_argument(
        "--volume_time", type=int, default=5, help="Volume measurement time in minutes"
    )
    argparser.add_argument(
        "--n_workers",
        type=int,
        default=3,
        help="Number of processes to use (if 1, no multi-processing)",
    )
    argparser.add_argument(
        "--rmax", type=int, default=100, help="Maximum range in kilometers"
    )
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
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # All sigmet files in inpath
    file_regex = args.file_regex.encode().decode("unicode_escape")
    files = utils.get_sigmet_file_list_by_task(args.inpath, file_regex=file_regex)

    filetime = lambda fn: utils.parse_time_from_filename(
        fn, file_regex, args.datetime_format, 0
    )

    # Get files matching the task
    prog = re.compile(args.task_regex.encode().decode("unicode_escape"))
    full_list = []
    for task in files.keys():
        result = prog.match(task)
        if result is None:
            continue

        full_list.extend(files[task])
    full_list = sorted(full_list)

    # Sort according to the time in files
    filetimes = [filetime(f) for f in full_list]
    timediffs = np.diff(sorted(filetimes))

    # Get volume intervals
    intervals = (
        pd.date_range(
            np.min(filetimes),
            np.max(filetimes) + timedelta(minutes=args.volume_time),
            freq=f"{args.volume_time}min",
        )
        .round(freq="T")
        .to_pydatetime()
    )

    volumes = []
    for i, (start, end) in enumerate(zip(intervals[:-1], intervals[1:])):
        vol = []
        for t, f in zip(filetimes, full_list):
            if t >= start and t < end:
                vol.append(f)
            elif t > end:
                break
        volumes.append(vol)

    # Drop empty volumes
    volumes = [v for v in volumes if len(v) > 0]

    plot_args = dict(
        path=args.inpath,
        outpath=args.outpath,
        quantities=args.qtys,
        rmax=args.rmax,
        range_rings_sep=args.range_rings,
    )

    if args.n_workers == 1:
        scheduler = "single-threaded"
    else:
        scheduler = "processes"

    # Run all volumes
    bag = db.from_sequence(volumes)
    bag.map(plot_volume_tasks, **plot_args).compute(
        num_workers=args.n_workers, scheduler=scheduler
    )
