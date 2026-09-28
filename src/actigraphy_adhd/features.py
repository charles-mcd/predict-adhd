"""Handcrafted features derived from the MCS daily activity aggregates.

Produces hand.csv: activity metrics for both the weekday/weekend split and
the non-split, standardised to active wear-time.
"""

# MCS daily aggregate columns, renamed to the study's activity intensity groups
RENAME = {'TOTPATY0': 'SED', 'TOTPATY1': 'LIT', 'TOTPATY2': 'MOD',
          'TOTPATY3': 'VIG', 'TOTCOUNT': 'CPM'}
ACTIVITY = ['SED', 'LIT', 'MOD', 'VIG', 'CPM']


def handcrafted(val, cov):
    """Build the handcrafted feature set for the valid sample."""
    # select activity and temporal features for valid sample
    df = val[['CMID', 'WKDAY', 'MTH', 'TTIMEDAY'] + list(RENAME)].rename(columns=RENAME)
    df = df[df.CMID.isin(cov.index)]

    # split
    wkday = df[df.WKDAY.eq('Yes')]
    weday = df[df.WKDAY.eq('No')]

    # get majority month and calc inter-daily standard deviation for SED and CPM
    # divide SED and COUNT by TotalTIME to standardise daily figures to active wear-time
    # sum activity metrics and divide by total wear time for period
    df[['SED_SD', 'CPM_SD']] = df[['SED', 'CPM']].div(df.TTIMEDAY, axis=0)

    df = df.groupby('CMID').agg(
        MTH=('MTH', lambda x: x.value_counts().idxmax()),
        SED_SD=('SED_SD', 'std'),
        CPM_SD=('CPM_SD', 'std'),
        TTIME=('TTIMEDAY', 'sum'),
        SED=('SED', 'sum'),
        LIT=('LIT', 'sum'),
        MOD=('MOD', 'sum'),
        VIG=('VIG', 'sum'),
        CPM=('CPM', 'sum')
    )

    df[ACTIVITY] = df[ACTIVITY].div(df.TTIME, axis=0)

    # sum weekday/weekend activity intensity categories and counts, divide by total wear mins
    # gives category proportions and avg. counts per min to standardise period figures
    wkday = wkday.drop(columns=['WKDAY', 'MTH']).groupby('CMID').sum()
    wkday[ACTIVITY] = wkday[ACTIVITY].div(wkday.TTIMEDAY, axis=0)

    weday = weday.drop(columns=['WKDAY', 'MTH']).groupby('CMID').sum()
    weday[ACTIVITY] = weday[ACTIVITY].div(weday.TTIMEDAY, axis=0)

    # merge for handcrafted featureset
    hand = (df.drop(columns='TTIME')
            .merge(wkday.drop(columns='TTIMEDAY'), on='CMID', suffixes=("", "_WK"))
            .merge(weday.drop(columns='TTIMEDAY'), on='CMID', suffixes=("", "_WE")))

    return hand
