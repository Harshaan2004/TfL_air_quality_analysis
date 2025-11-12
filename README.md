# TfL Air Quality Analysis

Statistical analysis of TfL air quality data focusing on the Northern Line as a worst-case scenario for pathogen transmission risk assessment. Part of dissertation: "Simulation and Machine Learning-Based Risk Assessment of Airborne Pathogen Transmission on the London Underground".

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Quick Start

1. **Clone or download the project**

   ```bash
   cd TfL_air_quality_analysis
   ```

2. **Set up virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis**
   ```bash
   python3 analysis.py
   ```

### Dependencies

- `pandas>=2.0.0` - Data manipulation and analysis
- `matplotlib>=3.7.0` - Data visualisation
- `scikit-learn>=1.3.0` - Machine learning and statistical modelling
- `numpy` - Numerical computations

## Usage

### Running the Analysis

```bash
python3 analysis.py
```

The script will:

- Load TfL air quality data
- Run statistical analysis with cross-validation
- Generate visualisation (`ventilation_vs_dust_plot_v2.png`)
- Output Northern Line analysis and CFD parameters
- Print all results to console

## Project Structure

```
TfL_air_quality_analysis/
├── analysis.py                    # Main analysis script
├── TfL_Analysis_Summary.md        # Detailed analysis summary
├── ventilation_vs_dust_plot_v2.png # Generated visualisation
├── README.md                      # This file
├── requirements.txt               # Python dependencies
└── venv/                          # Virtual environment (local)
```

## Files

- `analysis.py` - Main analysis script
- `TfL_Analysis_Summary.md` - Detailed findings and dissertation context
- `ventilation_vs_dust_plot_v2.png` - Generated visualisation
- `requirements.txt` - Python dependencies

## Notes

See `TfL_Analysis_Summary.md` for detailed results, findings, and dissertation integration strategy.
