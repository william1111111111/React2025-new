from reaction_flow.frd_shared import main
from .audit import OUT
if __name__=='__main__':main(root=OUT,models=('R0-baselined_step100','R1-calibrated_step100'))
