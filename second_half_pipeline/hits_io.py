"""Strict BLISS .dat and CSV schema (the upstream .dat header omits SEFD)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HIT_COLUMNS = [
    'Top_Hit_#', 'Drift_Rate', 'SNR', 'Uncorrected_Frequency',
    'Corrected_Frequency', 'Index', 'freq_start', 'freq_end',
    'SEFD', 'SEFD_freq', 'Coarse_Channel_Number', 'Bin_Width',
]
INTEGER_COLUMNS = ['Top_Hit_#', 'Index', 'Coarse_Channel_Number', 'Bin_Width']


def validate_hits(df, source):
    if list(df.columns) != HIT_COLUMNS:
        raise ValueError(f'{source}: expected the 12-column BLISS CSV schema including SEFD and Bin_Width; '
                         'regenerate legacy CSVs from their original .dat files')
    result = df.copy()
    for name in HIT_COLUMNS:
        result[name] = pd.to_numeric(result[name], errors='raise')
        values = result[name].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f'{source}: nonfinite values in {name}')
        if name in INTEGER_COLUMNS:
            if np.any(values < 0) or np.any(values != np.floor(values)) or np.any(values >= 2**63):
                raise ValueError(f'{source}: {name} must contain nonnegative integers')
            result[name] = result[name].astype('int64')
    return result


def read_dat(path):
    """Read all 12 row fields; skip comment lines, not a fixed header length.

    Fields 9/10 are SEFD placeholders. BLISS writes binwidth in field 12,
    despite labeling it Full_number_of_hits. Empty searches keep the schema.
    """
    rows = []
    saw_header = False
    with open(path) as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if line.lstrip('# ').startswith('Top_Hit_#'):
                    saw_header = True
                continue
            fields = line.split()
            if len(fields) != 12:
                raise ValueError(f'{path}:{line_number}: expected 12 BLISS fields, found {len(fields)}')
            rows.append(fields)
    if not saw_header:
        raise ValueError(f'{path}: missing BLISS column header (possibly incomplete output)')
    return validate_hits(pd.DataFrame(rows, columns=HIT_COLUMNS), path)


def read_hits_csv(path):
    return validate_hits(pd.read_csv(path), path)


def dat_to_csv(path, output):
    hits = read_dat(path)
    hits.to_csv(output, index=False)
    return hits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert BLISS .dat to correctly labeled 12-column CSV')
    parser.add_argument('dat_path')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.dat_path).resolve() == Path(args.output).resolve():
        parser.error('Output must differ from input')
    dat_to_csv(args.dat_path, args.output)
