"""Detached EMA gradient-ratio intervention; not the GradNorm algorithm."""
from dataclasses import dataclass, asdict
import math

LAMBDA_OLD=1.1338064670562744
@dataclass
class RatioController:
    A: float
    B: float
    completed: int=0
    lambda_current: float=LAMBDA_OLD
    target_ratio: float=1.0
    interval: int=20
    warmup: int=200
    decay: float=.9

    def __post_init__(self):
        self.validate_norms(self.A,self.B)

    @staticmethod
    def validate_norms(a,b):
        if not all(math.isfinite(x) and x>1e-12 for x in (a,b)):
            raise ValueError('nonfinite, empty or zero FM/task gradient norm')

    @property
    def raw_goal(self):return self.target_ratio*self.A/max(self.B,1e-12)
    @property
    def goal(self):return min(4*LAMBDA_OLD,max(.25*LAMBDA_OLD,self.raw_goal))
    @property
    def cap_hit(self):return self.raw_goal<=.25*LAMBDA_OLD or self.raw_goal>=4*LAMBDA_OLD
    @property
    def next_measurement(self):return (self.completed//self.interval+1)*self.interval

    def weight(self,arm,step):
        if step!=self.completed+1:raise ValueError('controller schedule mismatch')
        if arm=='B0-fixed':return LAMBDA_OLD
        if arm!='B1-ratio':raise ValueError('unknown arm')
        alpha=min(step/self.warmup,1.)
        return LAMBDA_OLD+alpha*(self.goal-LAMBDA_OLD)

    def advance(self,arm,step,norms=None):
        current=self.weight(arm,step)
        if step%self.interval==0:
            if norms is None:raise ValueError('measurement required')
            a,b=norms;self.validate_norms(a,b)
            self.A=self.decay*self.A+(1-self.decay)*a
            self.B=self.decay*self.B+(1-self.decay)*b
        elif norms is not None:raise ValueError('unexpected measurement')
        self.completed=step;self.lambda_current=current
        return self.weight(arm,step+1)

    def state(self,arm):
        return dict(**asdict(self),lambda_goal=self.goal,lambda_goal_raw=self.raw_goal,
                    cap_hit=self.cap_hit,next_measurement=self.next_measurement,
                    lambda_next=self.weight(arm,self.completed+1))

    @classmethod
    def restore(cls,state,arm):
        obj=cls(**{k:state[k] for k in cls.__dataclass_fields__})
        if obj.state(arm)!=state:raise ValueError('inconsistent persisted controller')
        return obj
