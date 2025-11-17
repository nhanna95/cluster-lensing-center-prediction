# Cluster Lensing Center Prediction

**This whole README.md was written by AI. There is no guarantee of accuracy of the information.**

Analysis toolkit for predicting galaxy cluster mass centers using gravitational lensing. This project fits hyperbolae and ellipses to quadruple lensed image systems (quads), computes geometric intersections to estimate cluster centers, and compares predictions with SZE and X-ray observations.

## Overview

This project implements a geometric approach to determine the mass center of galaxy clusters by analyzing gravitational lensing systems. When a background galaxy is lensed by a foreground cluster, it can produce multiple images (typically four, forming a "quad"). By fitting hyperbolae to these quads and finding their intersections, we can predict the cluster's mass center.

### Key Features

- **Hyperbola Fitting**: Fits hyperbolae to quadruple lensed image systems using geometric constraints
- **Ellipse Analysis**: Computes optimal ellipses through quad images for additional geometric constraints
- **Center Prediction**: Estimates cluster mass centers using weighted intersections of hyperbola pairs
- **FITS Processing**: Tools for processing and masking astronomical FITS images
- **Statistical Modeling**: Monte Carlo simulations for uncertainty quantification
- **Visualization**: Comprehensive plotting tools for quads, hyperbolae, intersections, and comparisons
- **Multi-Cluster Support**: Analyzes Abell 1689, CLJ2325, and other clusters

## Project Structure

```
.
├── center_prediction.py      # Main center prediction pipeline
├── FITS_mask.py              # FITS image masking and processing
├── FITS_to_jpg.py            # FITS to JPEG conversion
├── ellipse_diagrams.py       # Ellipse visualization
├── a1689_*.py                # Abell 1689 specific analysis scripts
├── stats_modeling/           # Statistical modeling and Monte Carlo simulations
├── util/                     # Core utilities
│   ├── data.py              # Data structures and cluster definitions
│   ├── math.py              # Geometric calculations (hyperbolae, ellipses, intersections)
│   ├── graphing.py          # Plotting utilities
│   └── coords_conversion.py # Coordinate transformations
├── fits/                     # FITS data files (not in repo)
├── figures/                  # Generated figures and plots (not in repo)
└── pkls/                     # Pickled data files (not in repo)
```

## Installation

### Dependencies

```bash
pip install numpy scipy sympy matplotlib plotly fitsio astropy pandas pillow tqdm
```

### Required Packages

- `numpy` - Numerical computations
- `scipy` - Scientific computing
- `sympy` - Symbolic mathematics for conic sections
- `matplotlib` - Plotting
- `plotly` - Interactive visualizations
- `fitsio` - FITS file I/O
- `astropy` - Astronomical utilities (WCS transformations)
- `pandas` - Data manipulation
- `Pillow` - Image processing
- `tqdm` - Progress bars

## Usage

### Basic Center Prediction

```python
# Run center prediction for a cluster
python center_prediction.py
```

The script will:
1. Load quad systems for the specified cluster
2. Fit hyperbolae to each quad
3. Compute pairwise intersections
4. Predict the cluster center using weighted averages
5. Generate visualization plots

### Configuration

Edit the configuration section in `center_prediction.py`:

```python
CLUSTER = du.clusters.get('a1689')  # or 'clj2325'

# Options
BOX_FILTER = True          # Restrict intersections to quad bounding box
WEIGHT_INTERS = True       # Weight intersections by fit error
PLOT_ELLIPSES = False      # Plot ellipses through quads
PLOT_ASYMS = True          # Plot hyperbola asymptotes
PLOT_INTERS = True         # Plot intersection points
```

### FITS Image Processing

```python
# Process FITS images and create masks
python FITS_mask.py

# Convert FITS to JPEG
python FITS_to_jpg.py
```

### Statistical Modeling

```python
# Run Monte Carlo simulations
python stats_modeling/a1689_stats_modeling.py

# Graph modeling results
python stats_modeling/graphing_modeling_results.py
```

## Methodology

### Hyperbola Fitting

For a quadruple lensed image system, the four images lie on a hyperbola. The hyperbola is fitted using geometric constraints based on the relative positions of the images.

### Center Prediction

1. **Hyperbola Generation**: Fit hyperbolae to each quad system
2. **Intersection Computation**: Find pairwise intersections between hyperbolae
3. **Filtering**: Optionally filter intersections to lie within quad bounding boxes
4. **Weighting**: Weight intersections by the inverse product of fit errors
5. **Averaging**: Compute weighted average of intersection points

### Validation

Predictions are compared against:
- **SZE (Sunyaev-Zel'dovich Effect)**: Thermal pressure center from SZE observations
- **X-ray**: X-ray brightness peak
- **Ellipse Centers**: Geometric centers of ellipses fitted through quads

## Data

FITS files and large data products are not included in this repository. To run the analysis, you'll need:

- FITS files in `fits/` directory
- Cluster quad definitions in `util/data.py`

## Examples

### Abell 1689 Analysis

```python
# Generate teaser figure
python a1689_teaser_fig.py

# Create comparison figures
python a1689_sze_fig.py
python a1689_xray_fig.py

# Generate insets
python a1689_insets.py
```

## Output

The analysis generates:
- **Figures**: Visualization plots in `figures/` directory
- **Masks**: Processed FITS masks in `fits/*/masks/`
- **Pickles**: Serialized results in `pkls/` directory

## Contributing

This is a research project. For questions or contributions, please open an issue or contact the maintainers.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite appropriately.

## Acknowledgments

Developed as part of an Undergraduate Research Opportunities Program (UROP) project on gravitational lensing analysis.

