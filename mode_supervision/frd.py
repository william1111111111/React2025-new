from reaction_flow.frd_shared import main
from .prepare import OUT
if __name__=='__main__':
    main(root=OUT,models=tuple(f'seed123_{arm}_step20000' for arm in ('M0-control','M1-mode')))
