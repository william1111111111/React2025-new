"""Reuse unchanged shared-prior evaluator; no repeated full shuffle diagnostic."""
from .balance_common import ROOT,ARMS
from .evaluate_shared import main
if __name__=='__main__':main(root=ROOT,allowed_arms=ARMS,allowed_steps=(16500,17000,18000),final_step=18000,final_shuffle=False)
