# Caffeine, Stress, and Sleep Mediation Analysis

## Project Overview

This repository accompanies a mediation analysis study examining whether perceived stress mediates the relationship between daily caffeine intake and sleep duration. The analytic framework follows the Baron-Kenny approach with bootstrap inference.

## Study Scope

- Independent variable: daily caffeine intake (mg)
- Mediator: perceived stress level
- Outcome: sleep duration (hours)
- Data source: synthetic health dataset used for reproducible statistical workflow demonstration

## Repository Contents

- `src/main.py`: end-to-end analysis pipeline
- `data/synthetic_coffee_health_10000.csv`: input dataset
- `figures/`: manuscript/report figures (JPEG)
- `requirements.txt`: Python dependencies
- `CITATION.cff`: citation metadata
- `LICENSE`

## Reproducibility

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/main.py
```

Outputs are generated under `Reports/`.

## Citation

Please cite this repository using `CITATION.cff`.

## License

This project is distributed under the MIT License.
