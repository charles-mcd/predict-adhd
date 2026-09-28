# Actigraphy and ADHD clinical prediction study

Code for predicting ADHD diagnosis from accelerometer data
recorded at age 7 in the UK Millennium Cohort Study (MCS). Code reproduced from an MSc thesis (University of York, 2026), available here: [Charlie McDermott's portfolio](https://charles-mcd.github.io)

- demonstrates data extraction and preprocessing of timeseries and survey data
- presents a reproducable machine learning pipeline for fair and interpretable modelling, following the TRIPOD+AI clinical prediction framework and satisfying certain NHS RAP criteria, [see  below](#reproducible-analytical-pipeline)
- examines whether intra-daily activity patterns and weekday/weekend differences improve prediction - relevant to condition phenotypes - and how many days of monitoring produce viable results.


## Data access

The MCS data are available to registered users from the UK Data Service, files from [data series SN: 2000031](http://doi.org/10.5255/UKDA-Series-2000031):

- Sweep 4 (age 7); accelerometer files and daily activity summary (SN: 7238)
- Parent interview files, sweeps 3 to 6; ADHD diagnosis question (SNs: 5795, 6411, 7464, 8156)
- Parent interview files, sweeps 1 and 4; covariates (SNs: 4683, 6411)


`scripts/prepare_data.py` reads these and produces three tables:

| File | Contents |
|---|---|
| `cov.csv` | ADHD class label, SEX, ETH, BMI, SDQ |
| `hand.csv` | handcrafted features from MCS daily aggregates |
| `auto.csv` | tsfresh features and a weekday flag |

Point `config.yaml` at the raw files (`data/MCS/`, including `dat/`); the
tables are written to `data/precomputed/`.

## Setup

Python 3.11+ and scikit-learn 1.8+.

```bash
pip install -r requirements.txt
pip install -e .
```

## Running


```bash
python scripts/prepare_data.py    # raw MCS files -> precomputed tables
python scripts/run_pipeline.py    # precomputed tables -> results
```

or `make data` and `make analysis`.

### Preparing the data

| Stage | Thesis section | Description | Cost |
|---|---|---|---|
| `cov` | 3.3.2 | ADHD labels across sweeps 3-6, covariates, exclusions | seconds |
| `hand` | 3.3.3 | Handcrafted features from the MCS daily aggregates | seconds |
| `ts` | 3.3.3 | `.dat` files to valid-day timeseries, non-wear flagging, day alignment | ~20 min, ~7GB |
| `tsfresh` | 3.3.3 | Feature extraction per participant-day | ~15 hours, ~10GB |
| `auto` | 3.3.3 | Concat the extraction chunks | minutes |

Run one stage with `--stage ts`. The tsfresh stage skips chunks that already
exist, so it can be stopped and resumed.


### Running the analysis

```bash
python scripts/run_pipeline.py                 # everything
python scripts/run_pipeline.py --stage final   # one stage
python scripts/run_pipeline.py --quick         # small bootstrap, separate results folder
```

| Stage | Thesis section | Description |
|---|---|---|
| `compare` | 4.1, 4.2 | Four feature conditions across four models, Wilcoxon test for RQ2, covariate models |
| `tuning` | B4 | Hyperparameter grid search |
| `final` | 4.3 | Out-of-fold predictions, ROC/PR curves, threshold analysis, coefficients |
| `subgroup` | 4.4 | Sex sensitivity analysis and calibration |
| `duration` | 4.5 | 2 to 6 day monitoring durations, paired bootstrap against 6 days |

Tables and figures are written to `results/`. The slow steps are saved to
`data/interim/` and reloaded on later runs; delete a file there to rebuild it,
or pass `--rebuild`. Cross-validation is 5-fold repeated 5 times.

## Layout

Based on [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)

```
config.yaml               paths and run options
Makefile                  make data, make analysis, make quick
pyproject.toml            package metadata
requirements.txt          requirements for reproducing env
scripts/
    prepare_data.py       raw MCS files -> precomputed tables
    run_pipeline.py       the analysis, in the order of the notebook
src/actigraphy_adhd/
    extract.py            labels, covariates, cohort exclusions
    features.py           handcrafted features from daily aggregates
    timeseries.py         .dat parsing, non-wear flagging, day alignment, tsfresh
    data.py               loading, feature conditions, duration day selection
    models.py             preprocessing, selectors, classifiers
    selection.py          feature reduction and selection
    evaluate.py           cross-validation, thresholds, subgroups, bootstrap
    plots.py              figures
```

## Reproducible Analytical Pipeline

Built to the NHS England [RAP Community of Practice](https://nhsdigital.github.io/rap-community-of-practice/introduction_to_RAP/levels_of_RAP/)
maturity framework, which the Goldacre review recommends as the minimum
standard for NHS and academic health data analysis.

**Baseline RAP**
- [x] Data produced by code in an open-source language (Python)
- [x] Code version controlled
- [x] README detailing the steps to reproduce
- [ ] Code peer reviewed
- [x] Code published in the open, linked from the write-up

**Silver RAP**
- [x] Outputs produced by code with minimal manual intervention
- [x] Documented: user guidance, structure, methodology, docstrings
- [x] Standard directory format
- [x] Reusable functions and classes
- [x] PEP8
- [ ] Testing framework (planned, v2.1.0)
- [x] Dependency information (`pyproject.toml`, `environment.yml`)
- [ ] Logs recorded automatically  (planned, v2.1.0)
- [x] Tidy data output


## Licence

MIT; see LICENSE.
