# ASTRA — Requirements Traceability Matrix

Requirement → design → implementation → verification. Updated at every gate.

| Req | Design | Implementation | Verification | Status |
|---|---|---|---|---|
| FR-01 | ARC §3.1, §4.1; SW-SDD §3.1 | `hardware/nvsmi.py`, `hardware/sysfs.py`, `hardware/probe.py` | TC-SW-01, 02, 12; TC-HW-04 | SW verified; HW pending |
| FR-02 | ARC §4.2; SW-SDD §3.1 | `validation/checks.py` L01–L10, `cli validate` | TC-SW-03, 04; TC-HW-05 | SW verified; HW pending |
| FR-03 | ADR-0003; SW-SDD §3.2 | `planner/split.py`, `planner/models.py` | TC-SW-06, 10; TC-RT-01 | SW verified; HW pending |
| FR-04 | ADR-0003, ADR-0004 | `planner/engines.py`, `cli plan/launch` | TC-SW-08; TC-RT-01, 02 | SW verified; HW pending |
| FR-05 | SW-SDD §3.3 | `validation/checks.py` R01–R05 | TC-SW-05; TC-RT-01, 02 | SW verified; HW pending |
| FR-06 | ADR-0007 | `telemetry/exporter.py` | TC-SW-09, 12 | Verified |
| FR-07 | ADR-0007; monitoring.md | `deploy/prometheus/astra-alerts.yml`, `deploy/grafana/` | TC-SW-14; TC-RT-04 | Rules verified (promtool); live pending |
| FR-08 | ADR-0008 | `deploy/compose/compose.yaml` profiles | TC-SW-14; TC-RT-01, 03 | Config verified; live pending |
| FR-09 | ARC §4.1 | `orchestration/ray_pool.py` | TC-SW-15; TC-RT-07 | SW verified; HW pending |
| FR-10 | ADR-0008 | `deploy/systemd/astra-selftest.service`, `astra-inference.service` | TC-RT-06; TC-HW-10 | Pending node |
| FR-11 | SW-SDD §4.1 | `cli --simulate` | TC-SW-06, 10 | Verified |
| FR-12 | SW-SDD §3.2 | `planner/gguf.py`, `models.from_gguf` | TC-SW-07 | Verified |
| FR-13 | ADR-0009; SW-SDD §3.2b | `hardware/gpu_specs.py`, `hardware/compat.py`, check L11, `astra compat` | TC-SW-16, 17; TC-HW-12 | SW verified; HW pending |
| FR-14 | ADR-0010; SW-SDD §3.2a | `planner/split.py` (`speed`, `select_gpus`) | TC-SW-19 | Verified |
| FR-15 | ADR-0009; HW-SDD §3.1 | `compat.power_budget`, check L12, exporter `astra_chassis_*`, alert `AstraChassisUndersized` | TC-SW-18; TC-HW-08 | SW verified; HW pending |
| FR-16 | ADR-0011 | `fabric/agent.py`, `fabric/discovery.py`, `fabric/protocol.py`, `astra agent`, `astra cluster`, `astra-agent.service` | TC-SW-20; TC-RT-08 | SW verified; multi-host pending |
| FR-17 | ADR-0011, ADR-0010 | `fabric/cluster.py`, `planner/split.py` (node hops), `planner/engines.py` (`--rpc`/`--device`) | TC-SW-21; TC-RT-08 | SW verified; multi-host pending |
| FR-18 | ADR-0011 | `cli._budgets` (GPU-less head) | TC-SW-22 | Verified |
| FR-19 | ADR-0012 | `astra/ui/server.py`, `astra/ui/static/index.html`, `astra/pool.py` | TC-SW-23, 24 | Verified (simulated + real GPU) |
| FR-20 | ADR-0013 | `astra/auto.py`, `hardware/nvtopo.py`, `astra/asbuilt.py` | TC-SW-25; TC-RT-10 | Verified (real PC) |
| FR-21 | ADR-0014 | `config.ModuleConfig`, `hardware/modules.py`, `compat.ModuleReport`, checks L12/L13, exporter `astra_module_*`, `asbuilt --stack`, console | TC-SW-29; TC-HW-13 | Verified in software (PC with 1 brick config); hardware pending |
| FR-22 | ADR-0014 | `planner/sizing.py`, `astra size` | TC-SW-27 | Verified |
| FR-23 | ADR-0015 | `planner/split.py` (draft), `planner/engines.py`, `astra/auto.py` (A/B) | TC-SW-26, TC-SW-28; TC-RT-12 | Verified (real PC) |
| NFR-01 | HW-SDD §2, §3.4 | L07, L08, L09, R05; alerts `AstraPcieReplays`, `AstraAerUncorrectable` | TC-HW-05; TC-RT-04 | Pending node |
| NFR-02 | HW-SDD §4 | L10, R04; alerts `AstraGpuHot`, `AstraGpuFaultSlowdown` | TC-HW-07; TC-RT-04 | Pending node |
| NFR-03 | HW-SDD §3.1, §3.3 | `setup-host.sh` step 4; alert `AstraChassisPowerHigh` | TC-HW-03, 08, 09 | Pending node |
| NFR-04 | HW-SDD §3.2, §3.4 | Assembly SOP | TC-HW-02, 11 | Pending node |
| NFR-05 | ADR-0001/0002; HW-SDD §2 | L05 | TC-HW-05, 06 | Pending node |
| NFR-06 | ADR-0005, ADR-0006 | stdlib core; CI OS matrix | TC-SW (CI matrix) | Verified (Windows locally; Linux in CI) |
| NFR-07 | ADR-0005; engineering standards | ruff, mypy strict, coverage gate | CI | Verified (90 % coverage, 171 tests) |
| NFR-08 | ADR-0007 | Collector rate limiting | TC-SW-09 | Verified |
| NFR-09 | SW-SDD §6 | systemd hardening, non-root image, localhost binds | Inspection at G6 | Pending review |
| NFR-10 | ADR-0008 | Restart policies; self-test gate | TC-HW-10; TC-RT-05, 06 | Pending node |
| NFR-12 | ADR-0011; deployment guide §10 | Firewall rules; read-only agent API | Inspection at G6 | Pending review |
| NFR-13 | ADR-0012 | localhost bind, fixed engine URL, 1 MiB body cap, inline assets | TC-SW-23 | Verified |
| NFR-11 | HW-SDD §1 | `hardware/bom.csv` | Analysis: €285–420 | Verified |

## Design-review findings → verification

| DR | Fixed in | Verified by |
|---|---|---|
| DR-01 | ADR-0003, `engines.vllm` | `test_vllm_launch_uses_pipeline_not_tensor_parallel`; TC-RT-02 |
| DR-02 | Planner fit check | `test_llama8b_fp16_does_not_fit_14gb_pool`, `test_llama8b_awq_fits_with_vllm` |
| DR-03 | Config default Gen3 x4 | `test_healthy_reference_node_passes` (L05) |
| DR-04 | BOM-03/04 | G3 BOM review; TC-HW-01 |
| DR-05 | setup-host.sh | TC-HW-09 |
| DR-07 | ADR-0004 | `test_llamacpp_launch` (UUID env); TC-RT-03 |
| DR-09 | L05 WARN logic | `test_uplink_idle_downshift_warns`; `test_parse_current_driver_schema` |
| DR-12 | BIOS checklist; L09 | `test_scan_classifies_signatures` (BAR) |
| DR-13 | Planner hint; ADR-0010 auto-selection | `test_single_gpu_hint_when_pooling_is_unnecessary`, `test_auto_leaves_slow_small_gpus_out_when_not_needed` |
| DR-16 | ADR-0009, L11 | `test_pascal_with_new_driver_fails`, `test_l11_l12_fail_for_pascal_on_new_driver_and_small_psu` |
| DR-17 | L12, compat | `test_two_3090s_overload_a_650w_psu` |
| DR-18 | ADR-0010 | `test_speed_gives_slow_gpu_only_the_spill` |
| DR-19 | compat, planner | `test_budget_mix_on_r580_needs_pinning_and_cuda12`, `test_vllm_auto_skips_pre_volta_and_explicit_use_is_rejected` |
