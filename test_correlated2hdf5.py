import numpy as np
from astropy.io import fits


uvfits_file = "/home/cat-work/work/swarmInterferometry/swarmData/3C286_0191C.UVF_1"

def decode_aips_baseline(bl):
    ant1 = bl // 256
    ant2 = bl % 256
    return ant1, ant2


with fits.open(uvfits_file, memmap=True) as hdul:

    # ============================================================
    # 1. General FITS inspection
    # ============================================================

    print("\n========== HDU SUMMARY ==========")
    hdul.info()

    hdu = hdul[0]
    hdr = hdu.header
    data = hdu.data

    print("\n========== PRIMARY HDU ==========")
    print("HDU type:     ", type(hdu))
    print("Data type:    ", type(data))
    print("Number groups:", len(data))

    print("\n========== FITS AXES ==========")

    for i in range(1, hdr.get("NAXIS", 0) + 1):
        print(
            f"Axis {i}: "
            f"NAXIS{i}={hdr.get(f'NAXIS{i}')}, "
            f"CTYPE{i}={hdr.get(f'CTYPE{i}')}, "
            f"CRVAL{i}={hdr.get(f'CRVAL{i}')}, "
            f"CRPIX{i}={hdr.get(f'CRPIX{i}')}, "
            f"CDELT{i}={hdr.get(f'CDELT{i}')}"
        )

    print("\n========== RANDOM GROUP PARAMETERS ==========")

    parnames = data.parnames
    print(parnames)

    for name in parnames:
        arr = np.asarray(data.par(name))

        print(
            f"{name:12s} "
            f"shape={arr.shape}, "
            f"dtype={arr.dtype}, "
            f"first values={arr[:5]}"
        )

    print("\n========== EXTENSIONS ==========")

    for i, ext in enumerate(hdul[1:], start=1):
        print(
            f"HDU {i}: "
            f"name={ext.name}, "
            f"type={type(ext).__name__}, "
            f"shape={getattr(ext.data, 'shape', None)}"
        )

    # ============================================================
    # 2. Inspect and store AIPS FQ table
    # ============================================================

    fq_hdu = next(
        (
            ext for ext in hdul[1:]
            if ext.name.strip().upper() == "AIPS FQ"
        ),
        None
    )

    if fq_hdu is None:
        raise ValueError("No AIPS FQ table found.")

    fq = fq_hdu.data

    print("\n========== AIPS FQ TABLE ==========")
    print("Columns:", fq.columns.names)
    print("Number of frequency setups:", len(fq))
    print("NO_IF:", fq_hdu.header.get("NO_IF"))

    for name in fq.columns.names:
        arr = np.asarray(fq[name])

        print(
            f"{name:20s} "
            f"shape={arr.shape}, "
            f"dtype={arr.dtype}"
        )

        print(arr)

    # This file has one frequency setup
    fq_row = fq[0]

    freq_ref_hz = float(hdr["CRVAL4"])

    if_freq_offset_hz = np.asarray(
        fq_row["IF FREQ"],
        dtype=np.float64
    )

    channel_width_hz = np.asarray(
        fq_row["CH WIDTH"],
        dtype=np.float64
    )

    total_bandwidth_hz = np.asarray(
        fq_row["TOTAL BANDWIDTH"],
        dtype=np.float64
    )

    sideband = np.asarray(
        fq_row["SIDEBAND"],
        dtype=int
    )

    # Since NAXIS4 = 1, each IF currently contributes one
    # frequency sample.
    if_freq_hz = freq_ref_hz + if_freq_offset_hz

    if_freq_mhz = if_freq_hz / 1e6
    channel_width_mhz = channel_width_hz / 1e6
    total_bandwidth_mhz = total_bandwidth_hz / 1e6

    print("\n========== FREQUENCY SUMMARY ==========")
    print(
        f"Primary reference frequency: "
        f"{freq_ref_hz / 1e6:.6f} MHz"
    )
    print(f"Number of IFs:               {len(if_freq_hz)}")
    print(
        f"IF frequency range:          "
        f"{if_freq_mhz.min():.6f} - "
        f"{if_freq_mhz.max():.6f} MHz"
    )

    print(
        "\n IF      Freq [MHz]    "
        "Chan width [MHz]    "
        "Total BW [MHz]    "
        "Sideband"
    )
    print(
        "------------------------------------------------"
        "-----------------------"
    )

    for i in range(len(if_freq_hz)):
        print(
            f"{i:3d}  "
            f"{if_freq_mhz[i]:12.6f}  "
            f"{channel_width_mhz[i]:16.6f}  "
            f"{total_bandwidth_mhz[i]:14.6f}  "
            f"{sideband[i]:8d}"
        )

    # ============================================================
    # 3. Inspect and store AIPS AN table
    # ============================================================

    an_hdu = next(
        (
            ext for ext in hdul[1:]
            if ext.name.strip().upper() == "AIPS AN"
        ),
        None
    )

    if an_hdu is None:
        raise ValueError("No AIPS AN table found.")

    an = an_hdu.data

    print("\n========== AIPS AN TABLE ==========")
    print("Columns:", an.columns.names)
    print("Number of antennas:", len(an))

    for name in an.columns.names:
        arr = np.asarray(an[name])

        print(
            f"{name:20s} "
            f"shape={arr.shape}, "
            f"dtype={arr.dtype}"
        )

        if arr.ndim <= 2 and arr.size < 100:
            print(arr)

    antenna_names = np.array([
        x.decode().strip()
        if isinstance(x, bytes)
        else str(x).strip()
        for x in an["ANNAME"]
    ])

    antenna_numbers = np.asarray(
        an["NOSTA"],
        dtype=int
    )

    antenna_xyz_m = np.asarray(
        an["STABXYZ"],
        dtype=np.float64
    )

    antenna_name_lookup = dict(
        zip(antenna_numbers, antenna_names)
    )

    print("\n========== ANTENNA SUMMARY ==========")

    for number, name, xyz in zip(
        antenna_numbers,
        antenna_names,
        antenna_xyz_m
    ):
        print(
            f"Antenna {number:3d}: "
            f"{name:12s} "
            f"XYZ = "
            f"[{xyz[0]:12.3f}, "
            f"{xyz[1]:12.3f}, "
            f"{xyz[2]:12.3f}] m"
        )

    # ============================================================
    # 4. Raw visibility data
    # ============================================================

    raw = np.asarray(data.data)

    print("\n========== RAW VISIBILITY ARRAY ==========")
    print("Shape:", raw.shape)
    print("dtype:", raw.dtype)

    raw = np.squeeze(raw)

    print("Shape after squeeze:", raw.shape)

    # For your current file:
    #
    # (nvis, nIF, npol, 3)
    #
    # Last dimension:
    #   0 = real
    #   1 = imaginary
    #   2 = weight

    if raw.ndim != 4:
        raise ValueError(
            "Expected squeezed DATA shape "
            "(nvis, nIF, npol, 3). "
            f"Instead found {raw.shape}."
        )

    nvis, nfreq, npol, ncomp = raw.shape

    if ncomp != 3:
        raise ValueError(
            "Expected final REAL/IMAG/WEIGHT axis "
            f"of length 3; found {ncomp}."
        )

    # ============================================================
    # 5. Construct complex visibilities
    # ============================================================

    vis = (
        raw[..., 0]
        + 1j * raw[..., 1]
    )

    weights = raw[..., 2]

    print("\n========== VISIBILITIES ==========")
    print("Complex vis shape:", vis.shape)
    print("Weight shape:     ", weights.shape)

    # vis is currently:
    #
    # [visibility_record, frequency/IF, polarization]

    # ============================================================
    # 6. Extract time and baseline values
    # ============================================================

    # Important:
    # Astropy automatically combines duplicate DATE random-group
    # parameters when accessed by name.
    jd = np.asarray(
        data.par("DATE"),
        dtype=np.float64
    )

    baseline = np.asarray(
        data.par("BASELINE"),
        dtype=int
    )

    integration_time_s = np.asarray(
        data.par("INTTIM"),
        dtype=np.float64
    )

    times = np.unique(jd)
    baselines = np.unique(baseline)

    print("\n========== TIME / BASELINE ==========")
    print("JD range:", jd.min(), jd.max())
    print("Unique times:", len(times))
    print("Unique baselines:", len(baselines))
    print("Baselines:", baselines)

    print(
        "Integration time range:",
        integration_time_s.min(),
        "-",
        integration_time_s.max(),
        "s"
    )

    # ============================================================
    # 7. Decode baselines and attach station names
    # ============================================================

    baseline_antennas = np.array(
        [
            decode_aips_baseline(bl)
            for bl in baselines
        ],
        dtype=int
    )

    baseline_names = np.array([
        [
            antenna_name_lookup.get(
                a1,
                f"ANT{a1}"
            ),
            antenna_name_lookup.get(
                a2,
                f"ANT{a2}"
            )
        ]
        for a1, a2 in baseline_antennas
    ])

    print("\n========== BASELINE SUMMARY ==========")

    for bl, ants, names in zip(
        baselines,
        baseline_antennas,
        baseline_names
    ):
        print(
            f"{bl:5d}: "
            f"{ants[0]:3d}-{ants[1]:3d}  "
            f"{names[0]} - {names[1]}"
        )

    # ============================================================
    # 8. Build [time, baseline, pol, freq] arrays
    # ============================================================

    nt = len(times)
    nb = len(baselines)

    dynamic_vis = np.full(
        (nt, nb, npol, nfreq),
        np.nan + 1j * np.nan,
        dtype=np.complex64
    )

    dynamic_weight = np.full(
        (nt, nb, npol, nfreq),
        np.nan,
        dtype=np.float32
    )

    time_index = {
        t: i
        for i, t in enumerate(times)
    }

    baseline_index = {
        b: i
        for i, b in enumerate(baselines)
    }

    for row in range(nvis):

        ti = time_index[jd[row]]
        bi = baseline_index[baseline[row]]

        # vis[row] is:
        #
        # [frequency, polarization]
        #
        # transpose to:
        #
        # [polarization, frequency]

        dynamic_vis[ti, bi, :, :] = vis[row].T
        dynamic_weight[ti, bi, :, :] = weights[row].T

    # ============================================================
    # 9. Final summary
    # ============================================================

    print("\n========== FINAL ARRAYS ==========")

    print(
        "Visibility [time, baseline, pol, freq]:",
        dynamic_vis.shape
    )

    print(
        "Weights    [time, baseline, pol, freq]:",
        dynamic_weight.shape
    )

    print("Time array [JD]:", times.shape)
    print("Frequency array [MHz]:", if_freq_mhz.shape)
    print("Baseline antenna pairs:", baseline_antennas.shape)
    print("Baseline station names:", baseline_names.shape)


### Peripheral Arrays:
# if_freq_hz
# if_freq_mhz

# channel_width_hz
# total_bandwidth_hz
# sideband

# antenna_numbers
# antenna_names
# antenna_xyz_m

# baseline_antennas
# baseline_names