# Plot radar files from RAW data

Some scripts to plot radar data from RAW files.

## Plot volumes

Plot radar volumes from all files in input directory matching the file and task regex, one volume (time interval) per file.

```bash
python plot_volumes.py <path-to-dir> <output-dir> --volume_time <how-many-minutes-is-one-volume> --task_regex <task-regex> --file_regex <file-regex> --rmax <max-distance-plotted> --datetime_format <date-format-in-file-regex> --n_workers <number-of-parallel-processes>
```

For example, plotting FMI volumes

```bash
python plot_volumes.py <path-to-dir> <output-dir> --volume_time 15 --task_regex "PPI[1-3]_[A-F]" --file_regex "([0-9]{12})_([A-Z0-9.\-_]+).raw" --rmax 250 --datetime_format "%Y%m%d%H%M" --n_workers 1
```

## Plot RHI scans in volumes

Plot radar RHI scans, a certain time interval in one file.

```bash
python plot_volume_rhis.py <path-to-dir> <output-dir> --volume_time <how-many-minutes-is-one-volume> --task_regex <task-regex> --file_regex <file-regex> --rmax <max-distance-plotted> --hmax <max-height-in-km> --datetime_format <date-format-in-file-regex> --n_workers <number-of-parallel-processes>
```

For example, plotting FMI RHIs

```bash
python plot_volume_rhis.py <path-to-dir> <output-dir> --volume_time 15 --task_regex "([A-Z\-_]*)(RHI)([A-Z\-_]*)" --file_regex "([0-9]{12})_([A-Z0-9.\-_]+).raw" --rmax 250 --hmax 25 --datetime_format "%Y%m%d%H%M" --n_workers 1
```

## Plot single PPIs with 4 quantities

```bash
python plot_radar_vars.py <path-to-raw-file> --out <out-directory> --qtys <ODIM-names-for-4-quantities> --rmax <max-distance-plotted> -hmax <max-height-in-km> --ext <pdf-or-png>
```

The script also allows plotting markers in the PPIS by specifying the `--markers` option. The markers are specified as a list of `lon,lat,marker_color,marker_type` values. The marker color and type should match available matplotlib options.

For example:

```bash
python plot_radar_vars.py <infile> --rmax 20 --qtys DBZH VRAD RHOHV ZDR --ext png
```

## Plot single RHIs with 4 quantities

```bash
python plot_radar_rhi.py <path-to-raw-file> --out <out-directory> --qtys <ODIM-names-for-4-quantities> --rmax <max-distance-plotted> --ext <pdf-or-png> --hmax <max-height-in-km>
```

For example:

```bash
python plot_radar_rhi.py <infile> --rmax 100 --hmax 25 --qtys DBZH VRAD RHOHV ZDR --ext png
```
