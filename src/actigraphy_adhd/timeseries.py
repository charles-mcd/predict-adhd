"""Raw actigraphy processing: .dat files to tsfresh features.

Produces auto.csv, one row per valid day per participant. The GT1M records
counts in 15-second epochs: a full day is 5760 epochs and the first
(part) day, recorded from 05:00, is 4560.

Runtimes and memory are substantial; see scripts/prepare_data.py.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

DAY_EPOCHS = 5760       # 00:00 - 23:59 at 15s epochs
FIRST_DAY_EPOCHS = 4560  # 05:00 - 23:59, when recording starts
NON_WEAR_RUN = 80       # >=80 consecutive zero epochs counts as non-wear
EXTREME_COUNT = 2928    # MCS substudy determines > 2928 is extreme, clipped


def dat_pandas(cmid, no_steps, dat_folder):
    """Import a .dat file into pandas and generate the non-wear flag."""
    raw = pd.read_csv(
        dat_folder / (cmid.lower() + '.dat'),
        skiprows=10,
        sep=r"\s+",
        header=None,
    )

    # certain .dat files spaced differently
    if cmid in no_steps:
        df = pd.DataFrame({
            "counts": raw.to_numpy().ravel(),
        }).dropna(how="all")

    else:
        df = pd.DataFrame({
            "counts": raw.iloc[:, 0::2].to_numpy().ravel()
        }).dropna(how="all")

    # flag for non-wear time: >=80 consecutive 0s
    s = df.iloc[:, 0].eq(0)                             # boolean for zero count
    groups = s.ne(s.shift()).cumsum()                   # groups for consec. true/false runs
    run_lengths = s.groupby(groups).transform('size')   # broadcast run length back to rows
    df["NW_flag"] = (s & (run_lengths >= NON_WEAR_RUN))  # flag rows 80+ zero-run

    return df


def get_offset(cmid, df, lookup):
    """Get the .dat file offset from the 1st valid day.

    Cycles through days of .dat data with non-wear flag until a match with
    MCS recorded weartime for first valid day. Returns offset in days or None.
    """
    # get wear from ts (flip non-wear bit)
    s = ~df.NW_flag

    # set first valid day & its wear count in epochs (mins * 4)
    val_day, val_wear = lookup.loc[cmid]
    val_wear = int(val_wear * 4)

    # initialise totals for first day
    win_wear = s[:FIRST_DAY_EPOCHS].sum()

    # return offset if window matches first valid day
    if win_wear == val_wear:
        return 0

    # check subsequent days
    n_days = (len(df) - FIRST_DAY_EPOCHS) // DAY_EPOCHS
    start = FIRST_DAY_EPOCHS
    end = FIRST_DAY_EPOCHS + DAY_EPOCHS

    for i in range(n_days):
        win_wear = s[start:end].sum()

        if win_wear == val_wear:
            return i + 1

        start += DAY_EPOCHS
        end += DAY_EPOCHS


def day_lookup(sw4_day):
    """Fast lookup for cmid to return idxs for each valid day and weekday."""
    def g(x):
        val_idx_raw = np.flatnonzero(x["val_bool"].to_numpy())

        if len(val_idx_raw) == 0:
            return pd.Series({"wkd_idx": [], "val_idx": []})

        offset = val_idx_raw[0]
        wkd_idx_raw = np.flatnonzero(x["wkd_bool"].to_numpy())

        return pd.Series({
            "wkd_idx": (wkd_idx_raw - offset).tolist(),
            "val_idx": (val_idx_raw - offset).tolist()
        })

    return (
        sw4_day.copy()
        .assign(
            wkd_bool=lambda x: x.WKDAY.eq("Yes"),
            val_bool=lambda x: x.VALIDDAY.eq("Yes")
        )
        .groupby("CMID")
        .apply(g)
    )


def build_timeseries(cov, sw4_day, dat_folder):
    """Prepare .dat files for tsfresh: valid days only, flagged and clipped.

    Runtime about 20 minutes, memory about 7GB.
    """
    # get list of recordings missing step count (affects parsing)
    no_steps = sw4_day[sw4_day.VALIDDAY.eq('Yes')].groupby('CMID').TOTSTEPS.first()
    no_steps = no_steps[no_steps == 0].index

    # quick lookup to get first valid day data
    lookup = sw4_day[sw4_day.VALIDDAY.eq('Yes')].groupby('CMID').first()[['TDATE', 'TTMEDAYE']]
    flag = day_lookup(sw4_day)

    dfs = []
    dropped_cmids = []

    for cmid in tqdm(cov.index):

        # convert .dat file to pandas with non-wear flag
        df = dat_pandas(cmid, no_steps, dat_folder)

        # extract timeseries offset (days before first valid day)
        offset = get_offset(cmid, df, lookup)

        # helper function approximates, testing reveals 0.2% IDs errorsome, drop
        if offset is None:
            dropped_cmids.append(cmid)
            continue

        # valid and wkday flags
        df['day'] = (df.index // DAY_EPOCHS) - offset
        df['valid'] = df.day.isin(flag.at[cmid, 'val_idx'])
        df['wkday'] = df.day.isin(flag.at[cmid, 'wkd_idx'])

        # select only valid days, drop now constant val col
        df = df[df.valid].drop(columns='valid')

        # add epochs and CMID
        df['epoch'] = range(len(df))
        df['CMID'] = cmid

        # handle extreme values
        df['counts'] = df['counts'].clip(lower=0, upper=EXTREME_COUNT)

        # add to list, concat after loop (saves multiple expensive ops)
        dfs.append(df)

    print(f'{len(dropped_cmids)} participants dropped, no offset match')
    return pd.concat(dfs, ignore_index=True), dropped_cmids


def day_ids(ts):
    """Create CMID_day IDs, including an identifier for wkday.

    Memory demand about 20GB.
    """
    tsF = ts.copy()
    tsF['CMID_DAY'] = (tsF['CMID'] + '_wkday:' + tsF['wkday'].astype(str)
                       + '_day:' + tsF['day'].astype(str))
    return tsF[['CMID_DAY', 'epoch', 'counts']]


def extract_chunks(tsF, chunk_folder, chunk_size=100, n_jobs=8):
    """tsfresh features per day per participant. Runtime about 15 hours.

    Written out in chunks so the run can be resumed; chunks already present
    are skipped.
    """
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters

    # list CMIDs, select compute efficient parameters
    cmids = tsF['CMID_DAY'].drop_duplicates().to_numpy()
    fc = EfficientFCParameters()

    chunk_folder.mkdir(parents=True, exist_ok=True)

    for chunk_num, start in enumerate(range(0, len(cmids), chunk_size)):
        path = chunk_folder / f'chunk_{chunk_num}.csv'

        if path.exists():
            print(f'chunk_{chunk_num}.csv exists, skipping')
            continue

        cmid_chunk = cmids[start:start + chunk_size]
        chunk_view = tsF[tsF["CMID_DAY"].isin(cmid_chunk)]

        tsf_features = extract_features(
            chunk_view,
            column_id='CMID_DAY',
            column_sort='epoch',
            column_value='counts',
            default_fc_parameters=fc,
            n_jobs=n_jobs               # EDA optimised number of concurrent workers
        )

        tsf_features.to_csv(path)


def concat_chunks(chunk_folder):
    """Concat the tsfresh chunks and recover the CMID and weekday flag."""
    auto = pd.DataFrame()

    for chunk in sorted(chunk_folder.iterdir()):
        auto = pd.concat([auto, pd.read_csv(chunk)])

    # extract CMID index and weekday flag from the CMID_DAY id
    auto['CMID'] = auto.iloc[:, 0].str[:9]
    auto['WKDAY'] = auto.iloc[:, 0].str[16].eq('T')
    auto = auto.iloc[:, 1:].set_index('CMID')

    print('tsfresh features:', auto.shape)
    return auto
