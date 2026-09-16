from reaction_flow.frd_shared import main
from .common import OUT,ARMS
if __name__=='__main__':main(root=OUT,models=tuple(a+'_'+v for a in ARMS for v in ['last-proposal','last-accepted']))
