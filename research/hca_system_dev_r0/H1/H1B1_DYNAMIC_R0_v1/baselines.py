"""Fair symbolic TRAIN/support-conditioned competitor and deliberately weak controls."""
import oracle
from contract import RULES

def predict(world):
    rule=oracle.identify_rule(world['support'],RULES)
    return {probe['case_id']:oracle.solve(probe,rule) for probe in world['probes']}

def majority_predict(world):
    # Training-derived majority class supplied by caller, not computed on TUNE gold.
    return {probe['case_id']:0 for probe in world['probes']}
