# ASTRA — Runbooks

General rules: **never unplug or replug the OCuLink cable with either system
powered.** Capture evidence before changing anything:

```bash
astra validate --format json --output /tmp/evidence-$(date +%s).json
journalctl -k -b --no-pager > /tmp/kernel-$(date +%s).log
```

## RB-01 GPU missing / driver down
*Alerts:* AstraGpuMissing, AstraExporterDown · *Checks:* L01, L02

1. `astra probe`. Is one GPU missing, or all of them?
2. **All missing, and the chassis fans are off:** the chassis lost power. Check the
   relay sense lead, the power strip and the chassis PSU switch. Ask whether the
   host went to sleep (`journalctl -b -1 | grep -i suspend`). Sleep must be masked
   (TC-HW-09).
3. **All missing, and the fans are on:** check `lspci -d 10b5:`. If the switch is
   absent, the uplink didn't train. Power both systems off, reseat the cable at both
   ends, and cold boot.
4. **One missing:** check `journalctl -k -b | grep -E "Xid|fallen off"`.
   Xid 79 ("fallen off the bus") → power off, reseat that GPU and its 8-pin, check
   the backplane power inputs, cold boot, then go to RB-03.
5. After recovery: `astra validate` must PASS. If it happens twice in 7 days, open
   a Sev-2 defect.

## RB-02 Degraded link
*Alert:* AstraUplinkDegraded · *Checks:* L05, L06

1. `astra probe` shows each hop's Gen/width. Find the hop below its expected
   value.
2. Uplink narrower than x4: power off, reseat the cable and the M.2 adapter, and
   check for cable damage or tight bends.
3. Uplink Gen below 3 **under load**: set the BIOS M.2 link to Gen3 (not Auto) and
   retest. If it persists, try the 0.5 m cable (DR-14).
4. GPU slot below its own maximum width (L06): reseat the GPU in its slot.

## RB-03 PCIe errors
Replays, AER or Xid. *Alerts:* AstraPcieReplays, AstraAerUncorrectable · *Checks:* L07, L08, L09, R05

1. Correlate: which GPU, and which hop (`astra_path_aer_errors_total` labels)? Did
   it coincide with load or temperature spikes?
2. Corrected errors only, rising slowly: schedule maintenance. Force Gen3 in BIOS
   and add `pcie_aspm=off` (`setup-host.sh --apply --disable-aspm`).
3. Uncorrectable errors, or Xid: **stop the inference workload**
   (`docker compose … stop` / `systemctl stop astra-inference`) to avoid corrupt
   results. Power off, reseat the cable and the GPUs, check earthing (TC-HW-11:
   both PSUs on the same strip), cold boot, and run the link gate plus a 1 h soak.
4. "BAR n: no space" at boot: enable Above 4G Decoding; add `--pci-realloc`.

## RB-04 Thermal
*Alerts:* AstraGpuHot, AstraGpuFaultSlowdown · *Checks:* L10, R04

1. Check the chassis fans are spinning, the intakes are unobstructed and the room
   ambient is ≤ 30 °C.
2. Short-term: cap power with `sudo nvidia-smi -i <idx> -pl <watts>` (e.g. 140 W on
   the 2060) and reduce concurrency.
3. Clean the dust filters and heatsinks. Confirm the GPU spacing is ≥ 1 slot.
4. `hw_power_brake_slowdown` means the PSU or backplane feed is sagging. Go to RB-05.

## RB-05 Power
*Alert:* AstraChassisPowerHigh

1. Sustained > 455 W means less than 30 % headroom on the 650 W PSU. Apply power
   limits (RB-04 step 2) or move a workload off the node.
2. Check that every backplane power input is connected (hardware design §3.1). A
   single 6-pin feed for two slots can brown out under load.

## RB-07 GPU swapped or added: missing, or L11/L12 fail
*Checks:* L02, L11, L12 · *Alert:* AstraChassisUndersized

1. `astra compat`. Read the driver window line and the findings.
2. **A card is absent from `nvidia-smi` after installing a GT 1030 or GTX 9xx/10xx:**
   the driver is newer than R580 and has dropped that architecture. Install the R580
   branch (`sudo apt install nvidia-driver-580`, or the matching `-server` package),
   reboot, and **hold it** (`sudo apt-mark hold nvidia-driver-580`). Every other GPU
   must also be supported by R580; `astra compat` confirms this.
3. **"no single driver supports every GPU":** remove the oldest card (Kepler cannot
   be mixed with anything current).
4. **L12 / AstraChassisUndersized:** fit the recommended PSU, or remove a card, or
   cap board power (`nvidia-smi -pl`) and set `chassis.psu_watts` accordingly.
5. **Engine fails with "no kernel image is available":** the engine build lacks the
   new card's architecture. Rebuild llama.cpp with
   `-DCMAKE_CUDA_ARCHITECTURES="<archs from astra compat>"` (CUDA 12.x if any
   pre-Turing card is present).
6. Regenerate the as-built config (`astra probe --emit-config`), re-plan
   (`astra plan … --output --env-file`), then run both gates (TC-HW-12).

## RB-06 Runtime split mismatch
*Check:* R02/R03 fail

1. Is the engine using a stale plan (a GPU was swapped, so its UUID changed)?
   Regenerate: `astra plan … --output plan.json --env-file …`, then
   `docker compose up -d`.
2. Did another process take VRAM after planning? `nvidia-smi` shows the processes.
   Re-plan: budgets use live free memory.
3. Consistently above the planned bytes (compute buffers): raise
   `planner.reserve_mib` in `/etc/astra/astra.toml` by 256 MiB and re-plan.
