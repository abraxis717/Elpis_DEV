"""R122 tests. Run explicitly (outside the repository CI lanes):

    python -B -m unittest discover -s research/yang_mills_hca/r122_energy_direction_forensics/tests -t .

Tests that need torch (the frozen teacher) skip with a reason when it is absent.
"""
