Introduction
------------
MERMAKE processes MERFISH, smFISH, and IF imaging data by detecting local intensity maxima (puncta) in 3D image stacks. In multiplexed experiments (MERFISH), these puncta are decoded using a user-supplied codebook, while in smFISH mode and IF, the puncta are simply called and reported as-is.

To install `MERMAKE`,
```
python3 -m pip install mermake
```

## ❗The newest version of mermake (>= 0.0.61) does drift but requires the nightly release of cupy available via:
 
```
pip install --pre cupy-cuda12x==14.0.0a1 -f https://pip.cupy.dev/pre
```

## ⚠️ GPU Requirements & CUDA Toolkit

MERMAKE relies on **CuPy** for GPU-accelerated image processing. To run MERMAKE successfully, you must have:

1. An **NVIDIA GPU** with CUDA support  
2. The **CUDA Toolkit** installed (Refer to the official CuPy installation [guide](https://docs.cupy.dev/en/stable/install.html))




##  MERMAKE Usage

To run **MERMAKE**, you'll need to provide a configuration TOML file with a few key settings describing your experiment. 
```
mermake my_settings.toml
```

If mermake is run without providing a toml file, it will warn about the usage and print out the toml file format. Most users only need to edit the `[paths]` section.

| TOML Section | Variable Name     | Description                                                                 |
|--------------|-------------------|-----------------------------------------------------------------------------|
| `[paths]`    | `codebook`        | Path to the CSV codebook for decoding barcodes (for MERFISH data only)     |
| `[paths]`    | `psf_file`        | Path to the PSF file used for deconvolution (e.g., `.npy` or `.pkl`)       |
| `[paths]`    | `flat_field_tag`  | Prefix path for flat field correction files (e.g., `"Scope3_"`)            |
| `[paths]`    | `hyb_range`       | Range of hybridization rounds to process (e.g., `'H1_*_set1:H1_*_set3'`) |
| `[paths]`    | `hyb_folders`     | List of folders containing raw imaging data                                |
| `[paths]`    | `output_folder`   | Path to the folder where MERMAKE should save results                        |


>  All other sections (`[hybs]`, `[dapi]`, etc.) are preconfigured for most use cases and usually do not need to be changed.  Though to use multi-psfs you will want to set the `tilesize` to the size of the samping grid (ie 300).

---

###  Example Config (`config.toml`)

```toml
[paths]
codebook = "codebook.csv"
psf_file = "psfs/psf_scope3.npy"
flat_field_tag = "flat_field/Scope3_"
hyb_range = "H1_*_set1:H16_*_set3"
hyb_folders = ["experiment_folder"]
output_folder = "output"
background_range = 'H0_background_set1:H0_background_set3'
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#           you probably dont have to change any of the settings below                  #
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#

hyb_save =  '{fov}--{tag}--col{icol}__Xhfits.npz'
dapi_save = '{fov}--{tag}--dapiFeatures.npz'
drift_save = 'drift_Conv_zscan__{ifov:0>3}--_set{iset}.pkl'
regex = '''([A-z]+)(\d+)_([^_]+)_set(\d+)(.*)''' #use triple quotes to avoid double escape

[hybs]
tile_size = 500
overlap = 89
beta = 0.0001
threshold = 3600
blur_radius = 30
delta = 1
delta_fit = 3
sigmaZ = 1
sigmaXY = 1.5

[dapi]
tile_size = 500
overlap = 89
beta = 0.01
threshold = 3.0
blur_radius = 50
delta = 5
delta_fit = 5
sigmaZ = 1
sigmaXY = 1.5



