"""Render RESULTS.md from an R122 result + tables pair (no model execution).

    python make_report.py RESULT.json TABLES.json OUT.md
"""
import json
import sys

res = json.load(open(sys.argv[1]))
tab = json.load(open(sys.argv[2]))
out = []
w = out.append
reg = res["mode"] == "REGISTERED_PROSPECTIVE"
ps = res["mechanism"]["scale"]
F = tab["strata"]["NATIVE"]["fields"]


def fx(x, d=4):
    return "n/a" if x is None else (f"{x:.{d}f}" if abs(x) < 1e4 else f"{x:.3e}")


def row(key, h="H20", pol="NATIVE"):
    return dict(zip(F, tab["strata"][pol]["strata"][key][h]))


w("# R122 — energy-normalized fidelity, directional safety and recurrent error localization")
w("")
w(f"**Mode:** `{res['mode']}`  ")
w(f"**Disposition:** `{res['disposition']}`  ")
w(f"**R93:** `{res['r93']}` (unchanged; a diagnostic cannot clear it)")
w("")
w(f"Protocol `{res['protocol_sha256']}`; source digest `{res['component_digest']}`; "
  f"raw tensor `{res['raw_tensor_sha256']}`; source commit `{res['source_guard']['git']['head']}`.")
w("")
w("## Integrity gates")
w("")
w("| Gate | Result |")
w("|---|---|")
for k, v in res["gates"].items():
    w(f"| {k} | {'PASS' if v is True else ('FAIL' if v is False else v)} |")
m, c = res["measurement"]["maxima"], res["measurement"]["counters"]
da, ta = res["decomposition_audit"], res["telescoping_audit"]
w("")
w(f"- Fixtures {c['fixtures']}; HCA passes {c['engine_steps']:,}; non-finite 0 = {c['nonfinite'] == 0}.")
w(f"- Composed teacher at k = 0 vs historical isolated teacher: max abs {m['teacher_k0_vs_historical_max_abs']:.1e} "
  f"({c['teacher_k0_bit_mismatch']} bit mismatches in {c['teacher_k0_checks']} step checks).")
w(f"- Independent torch-free teacher reference: max relative deviation {m['reference_teacher_max_rel']:.2e} ({c['reference_teacher_checks']} steps).")
w(f"- EXACT_GUARDED vs EXACT_R118: {c['exact_r118_mismatch']} mismatching trajectories of {c['exact_r118_checks']}; "
  f"fresh-state replay {c['fresh_state_mismatch']}/{c['fresh_state_checks']}; autonomous recheck after diagnostics "
  f"{c['autonomous_recheck_mismatch']}/{c['autonomous_rechecks']}; DELAYED8 H8 identity {c['delayed_h8_vs_native_mismatch']}/{c['delayed_checks']}.")
w(f"- Decomposition identities: max residual {da['max_identity_residual']:.2e}; exact rational audit "
  f"{m['exact_audit_max_relative']:.2e} over {c['exact_audit_samples']} samples; telescoping residuals "
  f"{ta['one_step_max_residual']:.2e} (one-step) and {ta['restart_max_residual']:.2e} (restart).")
g = res["gauge_controls"]
w(f"- Gauge: joint covariance {g['max_joint_covariance_abs']:.1e}; metric invariance {g['max_metric_invariance']:.1e}; "
  f"state-only negative control ≥ {g['min_state_only_negative_control']:.3f}; prediction-only half / quarter turns detected "
  f"{g['inconsistent_gauge_detected'][0]} / {g['inconsistent_gauge_detected'][1]} of {g['inconsistent_gauge_detected'][2]}.")
rf = res["regression_fixture"]
w(f"- Consumed R121 regression fixture: {rf['published_cells_compared']} published cell values reproduced "
  f"(worst relative deviation {rf['published_cells_worst_relative_deviation']:.1e}); inversion detected, extinguished predictions not reported as reversals.")
r = res["resources"]
w(f"- Resources: {r['seconds']} s, peak RSS {r['peak_rss_bytes'] / 2**20:.0f} MiB, raw tensor {r['raw_tensor_bytes'] / 2**20:.0f} MiB (volatile).")
nc = da["AUTO"]["norm_class_counts"]["READOUT"]
w(f"- READOUT autonomous samples by norm class: {json.dumps({k: v for k, v in nc.items() if v})}; "
  f"non-ordinary horizons: {json.dumps(da['readout_non_ordinary_by_horizon']) if da['readout_non_ordinary_by_horizon'] else 'none'}.")
w("")
w("## Registered mechanism decision (NATIVE, READOUT, H20, scale 1)")
w("")
dec = res["mechanism"]["decision"]
w("| Scope | DENOMINATOR_COLLAPSE | LOCAL_TRANSITION | ACCUMULATED_DRIFT | MIXED_LOCAL_ACCUMULATED | INVALID | dominant |")
w("|---|---:|---:|---:|---:|---:|---|")
for scope in ("ALL", "SIGNED", "UNSIGNED"):
    cnt = dec[scope]["counts"]
    w(f"| {scope} | {cnt['DENOMINATOR_COLLAPSE']} | {cnt['LOCAL_TRANSITION']} | {cnt['ACCUMULATED_DRIFT']} | "
      f"{cnt['MIXED_LOCAL_ACCUMULATED']} | {cnt['INVALID']} | {dec[scope]['dominant'] or 'none'} |")
w("")
w(f"Cohort dispositions: {json.dumps(res['cohort_dispositions'])}.")
w("")
w("Per graph class (strata = 3 overlaps × 2 cohorts):")
w("")
w("| Graph | " + " | ".join(["label counts"]) + " |")
w("|---|---|")
for gname, v in dec["ALL"]["by_graph"].items():
    w(f"| {gname} | {json.dumps(v)} |")
w("")
w("### Primary strata")
w("")
w("| Stratum | mechanism | E_rel mean | r | c | λ local | local E_rel | skill | A_den | state E_rel | state c | E_perp share | growth H8→H20 |")
w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for key, rec in tab["strata"]["NATIVE"]["strata"].items():
    if f"/{ps}/" not in key:
        continue
    h = row(key)
    gcls = rec["growth"]["class"]
    w(f"| {key.replace('/' + ps + '/', '/')} | {h['mechanism']} | {fx(h['erel_mean'])} | {fx(h['r_median'], 3)} | {fx(h['c_median'], 5)} | "
      f"{fx(h['lambda_local_median'], 3)} | {fx(h['local_erel_median'])} | {fx(h['local_skill_median'], 2)} | {fx(h['a_den_median'], 3)} | "
      f"{fx(h['state_erel_median'], 3)} | {fx(h['state_c_median'], 3)} | {fx(h['perp_share_median'], 4)} | {gcls} |")
w("")
w("Origin of the H20 READOUT error (median projection shares of the step-20 error by the block of steps whose local defects produced it; exact sum 1):")
w("")
w("| Stratum | steps 1–4 | 5–8 | 9–12 | 13–16 | 17–20 | Σ|share| |")
w("|---|---:|---:|---:|---:|---:|---:|")
for key, rec in tab["strata"]["NATIVE"]["strata"].items():
    if f"/{ps}/" not in key:
        continue
    o = rec["origin_READOUT"]
    w(f"| {key.replace('/' + ps + '/', '/')} | " + " | ".join(fx(x, 2) for x in o["share_median"]) + f" | {fx(o['absolute_share_sum_median'], 2)} |")
w("")
w("### By amplitude scale and by horizon (label counts over all strata)")
w("")
w("| Scale | DENOM | LOCAL | ACCUM | MIXED | INVALID | dominant | label agreement with scale 1 |")
w("|---|---:|---:|---:|---:|---:|---|---:|")
for s, v in res["mechanism_by_scale"].items():
    cnt = v["ALL"]["counts"]
    agree = res["amplitude_robustness"].get(s, {}).get("label_agreement_with_primary_scale")
    w(f"| {s} | {cnt['DENOMINATOR_COLLAPSE']} | {cnt['LOCAL_TRANSITION']} | {cnt['ACCUMULATED_DRIFT']} | {cnt['MIXED_LOCAL_ACCUMULATED']} | "
      f"{cnt['INVALID']} | {v['ALL']['dominant'] or 'none'} | {'—' if agree is None else fx(agree, 2)} |")
w("")
w("| Horizon (scale 1) | DENOM | LOCAL | ACCUM | MIXED | INVALID | SIGNED dominant | UNSIGNED dominant |")
w("|---|---:|---:|---:|---:|---:|---|---|")
for h, v in res["mechanism_by_horizon"].items():
    cnt = v["ALL"]["counts"]
    w(f"| H{h} | {cnt['DENOMINATOR_COLLAPSE']} | {cnt['LOCAL_TRANSITION']} | {cnt['ACCUMULATED_DRIFT']} | {cnt['MIXED_LOCAL_ACCUMULATED']} | "
      f"{cnt['INVALID']} | {v['SIGNED']['dominant'] or 'none'} | {v['UNSIGNED']['dominant'] or 'none'} |")
w("")
w("## Normalized-error growth H8 → H20 (R93 pattern = absolute defect falls while normalized defect rises)")
w("")
w("| Policy | Scale | Cohort | NO_GROWTH | NUMERATOR_DRIVEN | DENOMINATOR_DRIVEN | dominant |")
w("|---|---|---|---:|---:|---:|---|")
for pol, v in res["growth"].items():
    for s in v:
        for coh in ("SIGNED", "UNSIGNED"):
            cnt = v[s][coh]["counts"]
            if s == ps or pol == "NATIVE":
                w(f"| {pol} | {s} | {coh} | {cnt['NO_GROWTH']} | {cnt['NUMERATOR_DRIVEN']} | {cnt['DENOMINATOR_DRIVEN']} | {v[s][coh]['dominant'] or 'none'} |")
w("")
w("## Secondary findings (fixed criteria)")
w("")
for k, v in res["secondary_findings"].items():
    verdict = v.get("class", "SUPPORTED" if v.get("supported") else "NOT_SUPPORTED")
    w(f"- **{k}**: {verdict} (strata fraction {fx(v['strata_fraction'], 3)}; {v['criterion']}).")
lin = res["linear"]
w(f"- **S4 decayed-regime operator**: {'PASS' if lin['pass'] else 'FAIL'} — kernel probe is exactly linear (homogeneity "
  f"{lin['max_probe_homogeneity_abs']:.1e}); the eight-gate analytic operator equals it to {lin['max_analytic_vs_probe_rel']:.1e}; "
  f"linear-limit trajectories follow operator powers to {lin['max_linear_limit_trajectory_rel']:.1e}.")
w("")
w("Limiting gate values (zero invariants): teacher vs checkpoints")
w("")
w("| | k0 A←A | k1 A←sal | k2 A←col | k3 A←B | k4 B←B | k5 B←sal | k6 B←col | k7 B←A |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
w("| teacher | " + " | ".join(fx(x) for x in lin["teacher_gate_limits"]) + " |")
for ck, v in lin["checkpoint_gate_limits"].items():
    w(f"| {ck} | " + " | ".join(fx(x) for x in v) + " |")
w("")
w("Decayed-regime spectra relative to the teacher (over graph × k × checkpoint):")
w("")
w("| Cohort / operator | n | spectral-radius ratio median (min–max) | fraction decaying slower than teacher | dominant phase − teacher (°/step, median) | fraction with faster phase |")
w("|---|---:|---|---:|---:|---:|")
for k, v in lin["summary"].items():
    w(f"| {k} | {v['n']} | {fx(v['radius_ratio_median'])} ({fx(v['radius_ratio_min'])}–{fx(v['radius_ratio_max'])}) | "
      f"{fx(v['fraction_slower_decay_than_teacher'], 3)} | {fx(v['phase_minus_teacher_median_deg'], 3)} | {fx(v['fraction_faster_phase_than_teacher'], 3)} |")
w("")
w("## Directional safety (READOUT, all twenty steps)")
w("")
w("| Policy | in-range verdict | induced reversals in-range | above-clamp verdict | induced above-clamp | reversed samples by scale | induced by cohort |")
w("|---|---|---:|---|---:|---|---|")
for pol, d in res["directional"].items():
    w(f"| {pol} | {d['verdict_in_range_scales']} | {d['induced_in_range_all_steps']} | {d['verdict_above_clamp_scales']} | "
      f"{d['induced_above_clamp_all_steps']} | {json.dumps(d['reversed']['by_scale'])} | {json.dumps(d['induced_reversal']['by_cohort'])} |")
w("")
for pol, d in res["directional"].items():
    w(f"- {pol}: reversed by step {d['reversed']['by_horizon']}; induced by step {d['induced_reversal']['by_horizon']}; "
      f"induced by graph {json.dumps(d['induced_reversal']['by_graph'])}; first-reversal step histogram {json.dumps(d['first_reversal_horizon_histogram'])}.")
w("")
w("## Policy comparison at scale 1 (pooled READOUT NSE, equal-weight graph × checkpoint cells)")
w("")
pc = res["policy_comparison"]
w("| Overlap | Horizon | NATIVE | R114 | SIGNED_ONLY | DELAYED8 |")
w("|---|---|---:|---:|---:|---:|")
for key in sorted({k.split("/")[0] for k in pc["pooled"]}, key=lambda s: int(s[1:])):
    for h in (8, 20):
        w(f"| {key} | H{h} | " + " | ".join(fx(pc["pooled"][f"{key}/{p}/H{h}"], 6) for p in ("NATIVE", "R114", "SIGNED_ONLY", "DELAYED8")) + " |")
w("")
w("| Overlap / policy | mean ΔE_rel vs NATIVE (H20) | graph blocks improved | graph-cluster bootstrap 95 % | worst graph block |")
w("|---|---:|---:|---|---|")
for key, v in pc["paired_vs_native"].items():
    wkey = "/".join(key.split("/")[:2])
    wg = pc["worst_graph_harm_h20"][wkey]
    w(f"| {wkey} | {fx(v['mean_delta_erel'])} | {v['graph_blocks_improved']}/{v['graph_blocks']} | "
      f"[{fx(v['graph_cluster_bootstrap_ci95'][0])}, {fx(v['graph_cluster_bootstrap_ci95'][1])}] | {fx(wg['max_graph_block_delta_erel'])} ({wg['graph']}) |")
w("")
w(f"Worst graph-block harm over k: {json.dumps(pc['worst_graph_harm_h20_over_k'])}.")
ce = res["controller_effects"]
w("")
w("Controller-specific contributions:")
for coh, v in ce["teacher_forced_one_step"].items():
    w(f"- Teacher-forced one-step NSE is worse under the R114 step than under the NATIVE step in "
      f"{fx(100 * v['fraction_samples_local_nse_worse_under_r114'], 1)} % of {coh} samples (by scale {json.dumps(v['fraction_by_scale'])}).")
for key, v in ce["autonomous_step_effect"].items():
    w(f"- {key}: the controller's displacement raised the step error in {fx(100 * v['fraction_steps_error_raised_by_controller'], 1)} % "
      f"of controller-active steps (by scale {json.dumps(v['fraction_by_scale'])}).")
w("")
open(sys.argv[3], "w").write("\n".join(out) + "\n")
print("wrote", sys.argv[3], len(out), "lines")
