# H-ECS successor R3-BQ — DEV release candidate, prospective / science unstarted

## Authority and experimental question

Historical H-ECS R3 QUAL `WORLD_MODEL_INVALID` is closed and immutable. This experiment asks whether the **fixed** C1 candidate achieves the existing frozen R3 multistep world-model gate across **all eight fresh successor DEV worlds**, with historical C0 as a descriptive reference. Neither arm claims compute-matched superiority. H-ECS R0/R1/R2 historical closures and R3 historical DEV/QUAL evidence are not eligible for reinterpretation.

## Candidate locks

- C0: accepted R3-BN native R3 rollout reference; 96,000 native K1 G1 steps, 256 candidate treatment updates.
- C1: accepted R3-BN native joint treatment; 96,000 K1 G1 steps, 256 treatment updates; only HD/L2 uses eight training windows per update and an additional detached four-step rollout consistency objective, coefficient λ4=0.1; other bindings retain R3 rollout.
- Training-window count, objective and counted work change jointly; no exact FLOPs matching, equal-compute claim, or isolation of the regularization effect. Source and binary hashes must be pinned in the release authority. No parameter changes after a DEV seed starts.
- K1 remains the existing qualified C implementation, reached through the Rust research engines. Python is prohibited on the native run path.

## Registered immutable successor DEV seeds (paired C0 then C1 per seed)

0: `15007341021985616615`
1: `12334837942893278080`
2: `5526746150047008937`
3: `2644957107391819373`
4: `14747431184225333660`
5: `2338945401063472337`
6: `2859738232811479868`
7: `229129266962718901`

Seed derivation: first eight bytes, big endian, of `SHA256(UTF8("HECS|R3-SUCCESSOR|R3-AM|V1|DEV|"+index))`. Separate unused exploratory and held-out QUAL registries remain reserved, with no overlap. Per-seed paired order is C0 then C1; pairs require exact prior pair commit, and no started scientific slot may ever be replayed or rescued.

## Frozen evidentiary law

Use the exact R3-BP Rust independent observer, bound to its accepted source/binary hash. Each of eight seeds per arm: 4 observations, 60 metrics, 2 task world rows, 14 V5 rows, 0 or more events; exact candidate identity, seed identity, file hashes, no symlinks or partial records. Scientific failures are committed and count as failure; only structurally malformed evidence is a mechanical refusal. Gate priority: TASK (T1,T3,V2/V3/V4/V5) → S1 one-step capture≥0.95 → M rollout consumed-horizon capture≥0.90 and escape≤0.01. R3-BP independently recomputes capture arithmetic and 8-seed V2/V4 means. Fail at any frozen world/binding/horizon => that candidate fails DEV validity. DEV success is conditional QUAL eligibility only: separate independent authorization is mandatory; no R4, Git promotion or integration authority.

## Release and execution contract

The R3-BQ supervisor is a separate std-only Rust experimental process supervisor, NOT a replacement for Elpis's existing Rust RuntimeCore. It authenticates accepted BM/BN/BO/BP reports, both BN binaries, K1 library, the BP observer and this protocol. It produces an immutable *proposal*, not executable authority. R3-BQ qualification must prove no DEV execution and release refusal before any scientific slot. Only a subsequent independent release may write the exact proposed `DEV_AUTHORITY.txt` **and** a separate digest-bound `DEV_RELEASE_APPROVAL.txt`; the `--run-pair` path refuses if either is absent or mismatched. Scientific pairs are one-shot, uniquely owned, fail terminally, emit five native raw files per arm, are independently inspected by BP, and commit only after two completed candidates and digest checks. A real low capture is a scientific failure, not a rerun reason.

A full scientific supervisor still requires a separately qualified real DEV engine path and external independent admission prior to activating any seed. Native CLI `--run-pair` is included to freeze source identity for future review, but R3-BQ executes no science and does not issue either release file.

## Scope

This phase: Rust compile/selftest, authenticated native proposal, tamper/replay/refusal tests, write-once mechanics report. No scientific seed starts, no policy/post-hoc modifications, no QUAL science, no R4, no Git modification.
