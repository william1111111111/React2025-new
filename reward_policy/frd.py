from reaction_flow.frd_shared import main
from .common import OUT,ARMS
if __name__=='__main__':main(root=OUT,models=tuple(a+'_step500' for a in ARMS))
