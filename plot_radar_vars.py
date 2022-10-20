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
warnings.filterwarnings("ignore", category=RuntimeWarning)


def plot_fourpanel_fig(radar, qtys, max_dist=100, outdir=Path("."), ext="png"):
    cbar_ax_kws = {
        "width": "5%",  # width = 5% of parent_bbox width
        "height": "100%",
        "loc": "lower left",
        "bbox_to_anchor": (1.01, 0.0, 1, 1),
        "borderpad": 0,
    }

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(12, 10), sharex="col", sharey="row"
    )
    display = pyart.graph.RadarDisplay(radar)
    time = datetime.strptime(radar.time["units"], "seconds since %Y-%m-%dT%H:%M:%SZ")
    fmt = mpl.ticker.StrMethodFormatter("{x:.0f}")
    for ax, qty in zip(axes.flat, qtys):
        cax = inset_axes(ax, bbox_transform=ax.transAxes, **cbar_ax_kws)

        cmap, norm = utils.get_colormap(qty)
        # if norm is None:
        #     norm = mpl.colors.Normalize(vmin=qty_ranges[qty][0], vmax=qty_ranges[qty][1])
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
        cbar.set_label(label=utils.COLORBAR_TITLES[qty], weight="bold")

        if qty == "HCLASS":
            utils.set_HCLASS_cbar(cbar)

            # Set 0-values as nan to prevent them from being plotted with same color as 1
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

        for r in [20, 40, 60, 80, 250]:
            display.plot_range_ring(r, ax=ax, lw=0.5, col="k")
        # display.plot_grid_lines(ax=ax, col="grey", ls=":")
        ax.set_title(utils.TITLES[qty], y=-0.12)

    # x-axis
    for ax in axes[1][:].flat:
        ax.set_xlabel("Distance from radar (km)")
        ax.set_title(ax.get_title(), y=-0.22)
        ax.xaxis.set_major_formatter(fmt)

    # y-axis
    for ax in axes.flat[::2]:
        ax.set_ylabel("Distance from radar (km)")
        ax.yaxis.set_major_formatter(fmt)

    for ax in axes.flat:
        ax.set_xlim([-max_dist, max_dist])
        ax.set_ylim([-max_dist, max_dist])
        ax.set_aspect(1)
        ax.grid(zorder=15, linestyle="-", linewidth=0.4)

    # title = display.generate_title(field="reflectivity", sweep=0).split("\n")[0]
    fig.suptitle(
        f"{time:%Y/%m/%d %H:%M} UTC {radar.fixed_angle['data'][0]:.1f}°", y=0.02
    )

    fname = outdir / (
        f"radar_vars_{radar.metadata['instrument_name'].decode().lower()[:3]}_"
        f"{time:%Y%m%d%H%M%S}_{radar.fixed_angle['data'][0]:.1f}.{ext}"
    )

    # fig.subplots_adjust(wspace=0, hspace=0.2)
    fig.savefig(
        fname,
        dpi=600,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    # fig.savefig(f"radar_vars_{time:%Y%m%d%H%M}_{radar.fixed_angle['data'][0]:.1f}.png", dpi=600, bbox_inches="tight")


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
        "--ext",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Output image extension",
    )
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)
    plt.style.use("presentation.mplstyle")
    pyart.load_config(os.environ.get("PYART_CONFIG"))

    if len(args.qtys) != 4:
        raise ValueError("Wrong number of quantities, 4 expected!")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    infile = Path(args.inpath).resolve()

    radar = pyart.io.read_sigmet(infile)

    plot_fourpanel_fig(radar, args.qtys, max_dist=args.rmax, outdir=outdir)
