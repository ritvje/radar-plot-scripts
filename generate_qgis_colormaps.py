import numpy as np
import pyart
import utils
from matplotlib import cm, colors
from cmcrameri import cm as cm_crameri

# dl = 0.5
# cmapname = "pyart_RefDiff"
# color_levels = np.arange(-8, 8, dl)
# cmap, norm = utils.get_colormap("RHOHV")

# Scaled DBZH
dl_dbz = 0.5
scale = 0.01
offset = -327.68

original_levels = np.arange(-32, 70, dl_dbz)
scaled_levels = (original_levels - offset) / scale

norm = colors.BoundaryNorm(boundaries=scaled_levels, ncolors=len(scaled_levels))
cmap = cm.get_cmap("pyart_HomeyerRainbow", len(scaled_levels))

print(norm.boundaries)
# print(cmapname, color_levels[0], color_levels[-1])
with open(f"colormap_ODIM_scaled_DBZH.txt", "w") as f:
    f.write("# QGIS Generated Color Map Export File\DISCRETE:EQUAL INTERVAL")
    for i, level in enumerate(norm.boundaries):
        rgba = np.array(cmap(norm(level)))
        rgb = (rgba[:3] * 255).astype(int)
        #         print(f"{level},{rgb[0]},{rgb[1]},{rgb[2]},{int(rgba[3] * 100)},{level}\n")
        f.write(
            f"{level},{rgb[0]},{rgb[1]},{rgb[2]},{int(rgba[3] * 255)},{original_levels[i]} \n"
        )
