"""Extract the class labels, valid activity sample and covariates from the MCS files.

Produces cov.csv. Sweeps 3-6 parent interviews carry the ADHD diagnosis
question; sweep 4 carries the activity data and most covariates.
"""
from pathlib import Path

import pandas as pd

# ADHD diagnosis question by sweep: (file, cohort member number, question)
SWEEPS = {
    3: ('mcs3_parent_cm_interview.dta', 'CCNUM00', 'CPADHD00'),
    4: ('mcs4_parent_cm_interview.dta', 'DCNUM00', 'DPADHD00'),
    5: ('mcs5_parent_cm_interview.dta', 'ECNUM00', 'EPADHD00'),
    6: ('mcs6_parent_cm_interview.dta', 'FCNUM00', 'FPADHD00'),
}


def drop_bad_cats(path, bad_cols):
    """Parse .stata data into pandas.

    Stata allows duplicate categorical labels, pandas throws an error. This
    function drops the offending columns, extracts cat labels and reapplys
    them in pandas.
    """
    with pd.io.stata.StataReader(path) as reader:
        df = reader.read(convert_categoricals=False)
        value_labels = reader.value_labels()

    df = df.drop(columns=bad_cols, errors="ignore")

    for col, labels in value_labels.items():
        if col not in df.columns or col in bad_cols:
            continue

        # skip columns whose labels are not unique
        if len(set(labels.values())) != len(labels.values()):
            continue

        df[col] = pd.Categorical(df[col].map(labels).fillna(df[col]))

    return df


def parent_interviews(raw):
    """Load the sweep 3-6 parent interviews and add a Cohort Member ID."""
    pci = {}

    for sweep, (fname, cnum, _) in SWEEPS.items():
        df = pd.read_stata(raw / fname)
        df['CMID'] = df.MCSID + '_' + (df[cnum].cat.codes + 1).astype(str)
        pci[sweep] = df

    return pci


def adhd_ids(pci):
    """CMIDs with a reported ADHD diagnosis at any sweep."""
    positives = []

    for sweep, (_, _, question) in SWEEPS.items():
        df = pci[sweep]
        positives.append(df[df[question] == 'Yes'][['CMID', question]])

    adhd = positives[0]
    for df in positives[1:]:
        adhd = adhd.merge(df, on=['CMID'], how='outer')

    return adhd.CMID


def valid_days(raw):
    """Sweep 4 physical activity (PA) aggregates from the MCS study,
    limited to reliable recordings and valid days."""
    sw4_day = pd.read_stata(raw / 'mcs4_pa_main_activity_daily_data.dta')
    sw4_day['CMID'] = sw4_day.MCSID + '_' + sw4_day.DCNUM00.astype(str)
    val = sw4_day[sw4_day.RELIABLE.eq('YES') & sw4_day.VALIDDAY.eq('Yes')]

    return sw4_day, val


def class_labels(val, adhd):
    """Flatten the valid sample to one ID per row with class label."""
    labels = val.groupby('CMID', as_index=False)['CMID'].first()
    labels['ADHD'] = labels.CMID.isin(adhd)

    print('Total valid sample diagnosis distribution:\n', labels.ADHD.value_counts())
    return labels


def covariates(raw, pci, labels):
    """Merge the covariates onto the reliable CMIDs and apply the exclusions."""
    # sweep 4 predictor variables for ethnicity, BMI and parent-rated SDQ scores
    sw4_cmd = pd.read_stata(raw / "mcs4_cm_derived.dta")
    sw4_cmd['CMID'] = sw4_cmd.MCSID + '_' + (sw4_cmd.DCNUM00.cat.codes + 1).astype(str)

    sdq = sw4_cmd[['CMID', 'DDHYPER']].rename(columns={'DDHYPER': 'SDQ'})
    eth = sw4_cmd[['CMID', 'DDC06E00']].rename(columns={'DDC06E00': 'ETH'})

    sw4_cmi = drop_bad_cats(raw / "mcs4_cm_interview.dta", {"DPASSX0A"})
    sw4_cmi['CMID'] = sw4_cmd.MCSID + '_' + (sw4_cmd.DCNUM00.cat.codes + 1).astype(str)
    bmi = sw4_cmi[['CMID', 'DCBMIN4']].rename(columns={'DCBMIN4': 'BMI'})

    # sweep 1 predictor variable for SEX
    sw1_hhg = drop_bad_cats(raw / "mcs1_hhgrid.dta", {"AHINCA00"})
    sw1_hhg_CM = sw1_hhg[sw1_hhg.ACNUM00 != 0.0].copy()
    sw1_hhg_CM['CMID'] = sw1_hhg_CM.MCSID + '_' + sw1_hhg_CM.ACNUM00.cat.codes.astype(str)
    sex = sw1_hhg_CM[['CMID', 'AHCSEX00']].rename(columns={'AHCSEX00': 'SEX'})

    # sweep 4 confounding variables: autism/aspergers diagnosis and PA limited fitness
    sw4_pci, sw6_pci = pci[4], pci[6]

    fit = (sw4_pci[['CMID', 'DPCLSL00']]
           .loc[sw4_pci.DPCLSL00 == 'Yes']
           .rename(columns={'DPCLSL00': 'LIMFIT'}))

    aut = (sw4_pci[['CMID', 'DPAUTS00']]
           .loc[sw4_pci.DPAUTS00 == 'Yes']
           .rename(columns={'DPAUTS00': 'AUTISM'}))

    # sweep 6 gives ADHD medication status with age of onset
    # NICE guidelines suggest medication is not prescribed younger than 5,
    # ages up to 8 on medication confounding with PA, to be excluded
    med = (sw6_pci[(sw6_pci.FPMDLN00 > 4) & (sw6_pci.FPMDLN00 < 8)]
           [['CMID', 'FPMEDA00']]
           .rename(columns={'FPMEDA00': 'ONMEDS'}))

    cov = (labels
           .merge(sdq, on='CMID', how='left')
           .merge(eth, on='CMID', how='left')
           .merge(bmi, on='CMID', how='left')
           .merge(sex, on='CMID', how='left')
           .merge(fit, on='CMID', how='left')
           .merge(aut, on='CMID', how='left')
           .merge(med, on='CMID', how='left')
           ).set_index('CMID')

    # 24 rows found to be NaN for all covariates suggesting poor survey response, dropped
    cov = cov.dropna(subset=["ETH"])   # all 24 coincide with the only 24 NaNs in ETH

    # sex NaNs (N=191) converted to 'unknown'
    cov['SEX'] = cov['SEX'].cat.add_categories(['Unknown'])
    cov['SEX'] = cov['SEX'].fillna('Unknown')

    # SDQ converted from category to numeric
    cov.SDQ = pd.to_numeric(cov.SDQ, errors='coerce')

    # Ethnicity: only 7 non-white +ve class, pool
    cov['ETH'] = cov['ETH'].cat.add_categories(['Non-white'])
    cov.loc[cov.ETH != 'White', 'ETH'] = 'Non-white'

    # exclude confounders
    conf = ['AUTISM', 'LIMFIT', 'ONMEDS']
    cov = cov[~cov[conf].any(axis=1)].drop(columns=conf)

    print('after exclusions:', len(cov), 'participants,', int(cov.ADHD.sum()), 'ADHD')
    return cov


def with_dat_files(cov, dat_folder):
    """Drop participants whose raw .dat recording is absent."""
    dats = [file.name[:9].upper() for file in Path(dat_folder).iterdir()]
    cov = cov[cov.index.isin(dats)]

    print('with .dat recording:', len(cov), 'participants,', int(cov.ADHD.sum()), 'ADHD')
    return cov
