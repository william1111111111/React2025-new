"""Only new final B0/B1 endpoints; same fixed exact FRD20 implementation."""
from .balance_common import ROOT,ARMS
from .frd_shared import main
if __name__=='__main__':main(root=ROOT,models=tuple(f'seed123_{a}_step18000' for a in ARMS))
