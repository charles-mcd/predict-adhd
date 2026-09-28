"""Load the precomputed feature tables and build the feature conditions."""
import pandas as pd

# handcrafted feature sets, non-stratified and weekday/weekend stratified
HAND_NON = ['MTH', 'SED_SD', 'CPM_SD', 'SED', 'LIT', 'MOD', 'VIG', 'CPM']
HAND_STR = ['MTH', 'SED_SD', 'CPM_SD', 'SED_WK', 'LIT_WK', 'MOD_WK',
            'VIG_WK', 'CPM_WK', 'SED_WE', 'LIT_WE', 'MOD_WE', 'VIG_WE', 'CPM_WE']


def load_precomputed(folder):
    """Load from precomputed. Returns the unflattened auto_ for the duration study."""
    cov = pd.read_csv(folder / 'cov.csv', index_col='CMID')
    hand = pd.read_csv(folder / 'hand.csv', index_col='CMID')
    auto = pd.read_csv(folder / 'auto.csv', index_col='CMID')

    # save copy of unflattened auto for monitoring duration
    auto_ = auto.copy()

    # consistent sample (intersection of feature sets) for comparison testing
    final_idx = hand.join(auto, how='inner').join(cov, how='inner').index.unique()
    cov = cov.loc[final_idx]
    hand = hand.loc[final_idx]
    auto = auto.loc[final_idx]

    print(cov.ADHD.value_counts())

    return cov, hand, auto, auto_


def non_stratified(auto):
    """Average across all days."""
    return auto.groupby('CMID').mean()


def stratified(auto):
    """Stratify for weekdays/weekends."""
    auto_wk = auto[auto.WKDAY].groupby('CMID').mean().drop(columns='WKDAY')
    auto_we = auto[~auto.WKDAY].groupby('CMID').mean().drop(columns='WKDAY')
    return auto_wk.join(auto_we, how='inner', lsuffix='_WK', rsuffix='_WE')


def duration_sample(auto_):
    """Prep for duration study: participants with enough days of each type.

    Returns the day counts and the timeseries limited to those participants.
    """
    # get week/weekend days and total for each participant
    days = auto_.groupby('CMID').agg(
        COUNT=('WKDAY', 'count'),
        WKDAYS=('WKDAY', 'sum'),
        WEDAYS=("WKDAY", lambda x: (~x).sum())
    )

    # select >= 1 week/weekend day each and >= 6 total
    cmids = days[
        days.COUNT.ge(6) &
        days.WKDAYS.ge(1) &
        days.WEDAYS.ge(1)
        ].index

    df = auto_[auto_.index.isin(cmids)]

    print(f'duration sample: {df.index.nunique()} participants')
    return days, df


def _mean_days(df, n):
    """Average the first n days per participant, dropping the weekday flag."""
    return (df
        .groupby('CMID').head(n)
        .groupby('CMID').mean()
        .drop(columns='WKDAY'))


def duration_2(df, days):
    """Select 2 day duration."""
    # one week day
    df_wk = _mean_days(df[df.WKDAY], 1)

    # one weekend day
    df_we = _mean_days(df[~df.WKDAY], 1)

    return df_wk.join(df_we, how='inner', lsuffix='_WK', rsuffix='_WE')


def duration_3(df, days):
    """Select 3 day duration."""
    # two week days
    df_wk = _mean_days(df[df.WKDAY], 2)

    # one weekend day
    df_we = _mean_days(df[~df.WKDAY], 1)

    return df_wk.join(df_we, how='inner', lsuffix='_WK', rsuffix='_WE')


def duration_4(df, days):
    """Select 4 day duration."""
    # seperate based of availability
    mask = df.index.map(days["WEDAYS"].eq(1))
    df_m1 = df[mask]
    df_m2 = df[~mask]

    # for only 1 weekend day: three week days, one weekend day
    df_m1_wk = _mean_days(df_m1[df_m1.WKDAY], 3)
    df_m1_we = _mean_days(df_m1[~df_m1.WKDAY], 1)

    # for > 1 weekend day: two week days, two weekend days
    df_m2_wk = _mean_days(df_m2[df_m2.WKDAY], 2)
    df_m2_we = _mean_days(df_m2[~df_m2.WKDAY], 2)

    # concat availability splits and join week/weekend split
    df_wk = pd.concat([df_m1_wk, df_m2_wk])
    df_we = pd.concat([df_m1_we, df_m2_we])

    return df_wk.join(df_we, how='inner', lsuffix='_WK', rsuffix='_WE')


def duration_5(df, days):
    """Select 5 day duration."""
    # seperate based of availability
    mask = df.index.map(days["WEDAYS"].eq(1))
    df_m1 = df[mask]
    df_m2 = df[~mask]

    # for only 1 weekend day: four week days, one weekend day
    df_m1_wk = _mean_days(df_m1[df_m1.WKDAY], 4)
    df_m1_we = _mean_days(df_m1[~df_m1.WKDAY], 1)

    # for > 1 weekend day: three week days, two weekend days
    df_m2_wk = _mean_days(df_m2[df_m2.WKDAY], 3)
    df_m2_we = _mean_days(df_m2[~df_m2.WKDAY], 2)

    # concat availability splits and join week/weekend split
    df_wk = pd.concat([df_m1_wk, df_m2_wk])
    df_we = pd.concat([df_m1_we, df_m2_we])

    return df_wk.join(df_we, how='inner', lsuffix='_WK', rsuffix='_WE')


def duration_6(df, days):
    """Select 6 day duration."""
    # seperate based of availability
    mask_1 = df.index.map(days["WEDAYS"].eq(1))
    mask_2 = df.index.map(days["WKDAYS"].eq(3))

    df_m1 = df[mask_1]
    df_m2 = df[mask_2]
    df_m3 = df[~(mask_1 | mask_2)]

    # for only 1 weekend day: five week days, one weekend day
    df_m1_wk = _mean_days(df_m1[df_m1.WKDAY], 5)
    df_m1_we = _mean_days(df_m1[~df_m1.WKDAY], 1)

    # for only 3 weekdays: three week days, three weekend days
    df_m2_wk = _mean_days(df_m2[df_m2.WKDAY], 3)
    df_m2_we = _mean_days(df_m2[~df_m2.WKDAY], 3)

    # for the rest: four week days, two weekend days
    df_m3_wk = _mean_days(df_m3[df_m3.WKDAY], 4)
    df_m3_we = _mean_days(df_m3[~df_m3.WKDAY], 2)

    # concat availability splits and join week/weekend split
    df_wk = pd.concat([df_m1_wk, df_m2_wk, df_m3_wk])
    df_we = pd.concat([df_m1_we, df_m2_we, df_m3_we])

    return df_wk.join(df_we, how='inner', lsuffix='_WK', rsuffix='_WE')


# monitoring durations compared against the 6 day reference
DURATIONS = {2: duration_2, 3: duration_3, 4: duration_4,
             5: duration_5, 6: duration_6}
