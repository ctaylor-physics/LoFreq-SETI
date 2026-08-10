# FRB .venv for pipeline

---

### Problem: LSL 4.0.0 dropped all DP/ADP-era support

**Fix:** Created a separate venv `.venv-legacy-lsl` with LSL 3.0.8 instead of the system LSL 4.0.0.

bash

`python3 -m venv .venv-legacy-lsl
source .venv-legacy-lsl/bin/activate
pip install "numpy<2" "lsl==3.0.8" pandas`

---

### Patch 1: `spc_setup` attribute missing from `Session` class

**File:** `.venv-legacy-lsl/lib/python3.10/site-packages/lsl/common/sdf.py`

**Problem:** `hdfWaterfall.py` called `project.sessions[0].spc_setup` but LSL 3.0.8 only had `spcSetup` (camelCase).

**Fix:** Added a `spc_setup` property alias to the `Session` class:

python

`@property
def spc_setup(self):
    return self.spcSetup

@spc_setup.setter
def spc_setup(self, value):
    self.spcSetup = value`

---

### Patch 2: `get_mcs_hostname` missing from `metabundle`

**File:** `.venv-legacy-lsl/lib/python3.10/site-packages/lsl/common/metabundle.py`

**Problem:** `data.py` (called by `hdfWaterfall.py`) called `metabundle.get_mcs_hostname(tarball)` which doesn't exist in LSL 3.0.8. The older tarballs also lack an `ssmif.dat` file so `metabundle.get_station()` returns `None` and can't be used as a substitute.

**Fix:** Added `get_mcs_hostname` to `metabundle.py` backed by an environment variable, and added it to `__all__`:

python

`def get_mcs_hostname(tarname):
    station = __import__('os').environ.get('LWA_STATION_OVERRIDE')
    if station is None:
        raise RuntimeError(
            "get_mcs_hostname: LWA_STATION_OVERRIDE environment variable not "
            "set. Export it before running hdfWaterfall.py."
        )
    return station`

**Usage:** Always set this before running the pipeline:

bash

`export LWA_STATION_OVERRIDE=lwa1   # or lwasv, lwana`

---

### Patch 3: `asp_atten_3` missing from `get_asp_configuration()`

**File:** `.venv-legacy-lsl/lib/python3.10/site-packages/lsl/common/metabundle.py`

**Problem:** `data.py` expected `arx['asp_atten_3']` but older DP-era stations only had two attenuator stages (`asp_atten_1`, `asp_atten_2`) plus `asp_atten_split` — no third stage.

**Fix:** Wrapped the return value of `get_asp_configuration()` to inject a placeholder array of `-1`s if `asp_atten_3` is absent:

python

`result = backend.get_asp_configuration(tarname, which=which)
if 'asp_atten_3' not in result:
    ref_key = 'asp_atten_1' if 'asp_atten_1' in result else next(iter(result))
    result['asp_atten_3'] = [-1] * len(result[ref_key])
return result`

---

### Patch 4: `asp_atten_3` missing from `get_asp_configuration_summary()`

**File:** `.venv-legacy-lsl/lib/python3.10/site-packages/lsl/common/metabundle.py`

**Problem:** Same as Patch 3 but `data.py` was actually calling the summary version of the function, which returns scalar values rather than arrays.

**Fix:** Same injection but with a scalar `-1`:

python

`result = backend.get_asp_configuration_summary(tarname, which=which)
if 'asp_atten_3' not in result:
    result['asp_atten_3'] = -1
return result`

---

### Patch 5: `mcs.host` missing from older tarballs

**File:** `final_scripts/get_metadata.py` (your own code)

**Problem:** `get_station()` tried to read `mcs.host` from the tarball to determine the station, but older tarballs don't contain this file.

**Fix:** Added a `station` override parameter so the station can be passed in from the command line:

python

`def get_station(filename, station=None):
    if station is not None:
        return station.lower()
    with tarfile.open(filename) as tar:
        station = tar.extractfile('mcs.host').read().decode().strip().replace('-tp', '')
    return station`

**Usage:** Always pass `--station` when running the pipeline on older data:

bash

`python3 -m final_scripts.pipeline --tar-path '...' --drx-path '...' \
    --avg 3.0 --length 8388608 --station lwa1`

---

### Summary: full invocation for older data

bash

`source ~/scripts_lillian_ucf3/.venv-legacy-lsl/bin/activate
export LWA_STATION_OVERRIDE=lwa1   # or lwasv / lwana
python3 -m final_scripts.pipeline \
    --tar-path '/path/to/DD002_XXXX.tgz' \
    --drx-path '/path/to/drx_seti' \
    --avg 3.0 --length 8388608 \
    --station lwa1`

All patches are isolated to `.venv-legacy-lsl` and `get_metadata.py` — the shared scripts (`hdfWaterfall.py`, `data.py`) were never touched.
