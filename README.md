# Elpis (development foundation)

Elpis is a research system organised around one idea, *geodesic intelligence*:
persistent structured state, memory topology, bounded transformations,
controlled evolution and inference should interact so that computation
follows structure the system has already accumulated.

This repository is the clean development authority. It is being built by a
selective, audited migration from the Elpis beta repository. The migration
record is [`migration/BETA_MIGRATION.json`](migration/BETA_MIGRATION.json), and
the reasoning is in [`docs/MIGRATION.md`](docs/MIGRATION.md).

The single machine-readable system authority is
[`ELPIS_SYSTEM.json`](ELPIS_SYSTEM.json).

```bash
python -m pip install ".[test]"
python -m pytest
```
