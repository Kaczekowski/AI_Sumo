"""Portable FSM and zero-order Sugeno fuzzy controllers. Only standard library."""
import json
import math
import random
from pathlib import Path
from physics import Action, Config, Robot, move, norm

# Bounds are also the genome schema. Each saved model contains actual learned values.
BOUNDS = dict(edge=(0.55,0.9),close=(0.18,0.5),flank=(0.18,0.95),
              power=(0.65,1.0),safety=(0.5,4.0),pressure=(0.2,3.0),
              approach=(0.2,2.0),width=(0.06,0.25),hysteresis=(0.02,0.12))
DEFAULT = dict(edge=.78,close=.32,flank=.5,power=.95,safety=2.0,
               pressure=1.5,approach=.8,width=.14,hysteresis=.06)

def ramp(x, center, width):
    return max(0.0,min(1.0, .5+(x-center)/(2*width)))

class Controller:
    def __init__(self, kind="fsm", params=None):
        if kind not in ("fsm","fuzzy"): raise ValueError("kind must be fsm or fuzzy")
        self.kind=kind
        self.p=dict(DEFAULT if params is None else params)
        if set(self.p)!=set(BOUNDS): raise ValueError("Incomplete parameter schema")
        for k,(lo,hi) in BOUNDS.items():
            if not math.isfinite(self.p[k]) or not lo <= self.p[k] <= hi:
                raise ValueError("Invalid parameter: "+k)
        self.reset()

    def reset(self):
        self.state="APPROACH"
        self.last_distance=None
        self.stalled=0
        self.rng=None

    def act(self, me, enemy, arena, cfg=Config()):
        p=self.p
        if self.rng is None:
            # Local mixed strategy breaks mutual blocking, reproducible from the start.
            self.rng=random.Random(round(me.x*1000)*1000003+round(me.y*1000))
        d=math.hypot(enemy.x-me.x,enemy.y-me.y)
        edge=(math.hypot(me.x,me.y)+me.radius)/arena
        close=d/arena
        self.stalled = self.stalled+1 if self.last_distance is not None and abs(d-self.last_distance)<8 else 0
        self.last_distance=d
        if self.kind=="fsm":
            threshold=p["edge"]-(p["hysteresis"] if self.state=="RECOVER" else 0)
            if edge>threshold: self.state="RECOVER"
            elif self.stalled>=1: self.state="FLANK"
            elif close<p["close"]: self.state="PUSH"
            else: self.state="APPROACH"
            safety=p["safety"]*(2 if self.state=="RECOVER" else .65)
            pressure=p["pressure"]*(1.5 if self.state=="PUSH" else 1)
            approach=p["approach"]
            flank=p["flank"] if self.state=="FLANK" else 0.0
        else:
            # Four fuzzy rules; product AND; normalized Sugeno consequents.
            danger=ramp(edge,p["edge"],p["width"])
            near=1-ramp(close,p["close"],p["width"])
            stuck=min(1.0,self.stalled/2)
            near=max(near,stuck)
            weights=[danger,(1-danger)*near*(1-stuck),
                     (1-danger)*(1-near),(1-danger)*near*stuck]
            # consequents: safety multiplier, pressure multiplier, flank preference
            outputs=[(2.5,.5,0),(.55,1.7,0),(.8,1,0),(.8,1.2,p["flank"])]
            total=sum(weights)
            safety=p["safety"]*sum(w*v[0] for w,v in zip(weights,outputs))/total
            pressure=p["pressure"]*sum(w*v[1] for w,v in zip(weights,outputs))/total
            flank=sum(w*v[2] for w,v in zip(weights,outputs))/total
            approach=p["approach"]
            self.state="FUZZY_"+["RECOVER","PUSH","APPROACH","FLANK"][max(range(4),key=weights.__getitem__)]
        heading=math.atan2(enemy.y-me.y,enemy.x-me.x)
        center=math.atan2(-me.y,-me.x)
        # Compact one-turn model-based action selection, common to both AI families.
        # Opponent prediction is a simple direct attack, never its actual hidden command.
        predicted=Action(heading+math.pi,1)
        angles=[heading,heading+p["flank"],heading-p["flank"],
                heading+1.15,heading-1.15,center]
        candidates=[Action(a,power,self.state) for a in angles for power in (p["power"],.5,.15)]
        best=None
        for c in candidates:
            a,b,w,hit,_=move(me,enemy,c,predicted,arena,cfg)
            if w is None:
                out_a=math.hypot(a.x,a.y)+a.radius>arena*cfg.shrink
                out_b=math.hypot(b.x,b.y)+b.radius>arena*cfg.shrink
                if out_a or out_b: w=-1 if out_a and out_b else (1 if out_a else 0)
            ra=math.hypot(a.x,a.y)/arena
            rb=math.hypot(b.x,b.y)/arena
            gap=math.hypot(a.x-b.x,a.y-b.y)/arena
            outcome=100 if w==0 else (-100 if w==1 else (-25 if w==-1 else 0))
            # Penalize next shrink boundary, not merely the current one.
            risk=max(0,(math.hypot(a.x,a.y)+a.radius)/(arena*cfg.shrink)-.85)
            offset=abs(math.atan2(math.sin(c.angle-heading),math.cos(c.angle-heading)))
            value=(outcome+pressure*rb-safety*(ra*.25+5*risk)
                   -approach*gap+4*flank*min(offset,1.15)+.03*c.power)
            if self.stalled:
                value+=self.rng.uniform(-2,2)*min(self.stalled,2)
            if best is None or value>best[0]: best=(value,c)
        return best[1]

    def save(self,path,metadata=None):
        Path(path).write_text(json.dumps(dict(schema=1,kind=self.kind,parameters=self.p,
                                             metadata=metadata or {}),indent=2),encoding="utf-8")

def load(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema")!=1: raise ValueError("Unsupported model schema")
    return Controller(data["kind"],data["parameters"])

class Baseline:
    def __init__(self,kind="charge"): self.kind=kind
    def reset(self): pass
    def act(self,me,enemy,arena,cfg=Config()):
        angle=math.atan2(enemy.y-me.y,enemy.x-me.x)
        if self.kind=="center" and math.hypot(me.x,me.y)>.55*arena:
            angle=math.atan2(-me.y,-me.x)
        return Action(angle,1,"BASELINE")
