# gimvi-reproducibility
Reproducibility of gimVI manuscript [ADD LINK]()

## Analysis of the article
The article contains the comparison of two pairs of datasets for which the analysis can be found in the following notebook links:
- [Pair *F*: osmFISH and scRNA-seq](gimvi_reproduce_osm_fish.ipynb)
- [Pair *S*: starMAP and scRNA-seq](gimvi_reproduce_starmap.ipynb)

## Files in the repo

The model gimVI is implemented in the [scVI repo](https://github.com/YosefLab/scVI) as a JVAE model (Joint Variational AutoEncoder).

Every model we benchmark gimVI against are encapsulated into Benchmarkable objects (as well as gimVI itself), found in [benchmarkable.py](benchmarkable.py). The metrics are all grouped into [metrics.py](metrics.py).
In order to perform the benchmark with Seurat an Liger written in R, we implemented R-Python bridges in the following files:
- [Seurat.functionsV3.R](Seurat.functionsV3.R) for Seurat R functions
- [seuratV3.py](seuratV3.py) for the Seurat Python object
- [liger.py](liger.py) for the liger object (R function are directly called from it)

Finally [plot_benchmarkable.py](plot_benchmarkable.py) contains some helper function for additionnal plots in the notebooks.
