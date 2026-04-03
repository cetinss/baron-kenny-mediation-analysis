# Public Reproducibility Repository

This repository is a public reproducibility package for the JIENS mediation analysis.
It is intended for reviewers, researchers, students, and anyone who wants to inspect or rerun the analysis pipeline.

## Included files

- `src/main.py`: main Baron-Kenny mediation pipeline
- `data/synthetic_coffee_health_10000.csv`: synthetic dataset used in analysis
- `figures/`: exported JPG figures used in the final report/manuscript workflow
- `requirements.txt`: Python dependencies
- `LICENSE`

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/main.py
```

## Output

The scripts write analysis tables and figures under `Reports/`.

Note: The manuscript text was written manually by the authors; `ARTICLE_TEXT.txt` generation is intentionally excluded from this repository.

## Citation

Use the entry in `CITATION.cff` or the BibTeX below.

```bibtex
@software{cetin_oztoprak_2026_caffeine_stress_sleep,
  author = {Cetin, Sena and Oztoprak, Elif Beyza},
  title = {Public Reproducibility Repository: Caffeine, Stress, and Sleep Mediation Analysis},
  year = {2026},
  url = {https://github.com/cetinss/baron-kenny-mediation-analysis}
}
```
