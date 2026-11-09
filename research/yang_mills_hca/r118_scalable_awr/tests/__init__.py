"""R118 qualification tests. Run explicitly (not part of the repository CI lanes):

    python -B -m unittest discover -s research/yang_mills_hca/r118_scalable_awr/tests -t .

Tests that need torch (the frozen oracle) skip with an explicit reason when torch
is absent; every other test is numpy + stdlib only.
"""
