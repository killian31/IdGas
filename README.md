# IDGAZ Project

Machine learning models for gas detection using sensor data.

## Project Structure

```
├── *.ipynb                      # Notebooks in root directory
├── idgaz/                       # Main package
│   ├── models/                  # Model implementations
│   │   ├── linear.py            # Linear model implementations
│   │   ├── tree_based.py        # Random forest and XGBoost models
│   │   └── neural/              # Neural network models
│   │       ├── architectures.py # Network architectures
│   │       ├── losses.py        # Custom loss functions
│   │       ├── training.py      # Training utilities
│   │       └── uda.py           # Unsupervised domain adaptation
│   ├── data/                    # Data processing
│   │   ├── preprocessing.py     # Data preprocessing
│   │   └── feature_engineering.py # Feature engineering
│   └── utils/                   # Utilities
│       ├── evaluation.py        # Metrics and evaluation
│       ├── visualization.py     # Plotting and visualization
│       └── io.py                # Data loading/saving
```

## Usage

Import modules in notebooks using:

```python
from idgaz.models import tree_based
from idgaz.data import preprocessing
```
