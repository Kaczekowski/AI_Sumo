
# physics.py

from dataclasses import dataclass, replace, asdict
import math
import random

def norm(x, y):
    d = math.hypot(x, y)
    return (x/d, y/d) if d > 1e-12 else (1.0, 0.0)

@dataclass(frozen=True)
class Config:
    arena: float = 900.0
    shrink: float = 0.86
    rounds: int = 10
    distance: float = 200.0
    min_distance: float = 50.0
    push_reward: float = 2.0
    stall_penalty: float = 0.5
    restitution: float = 0.25
    drag: float = 0.7

@dataclass(frozen=True)
class Robot:
    x: float
    y: float
    mass: float = 1.0
    radius: float = 50.0
    drive: float = 1.0

@dataclass(frozen=True)
class Action:
    angle: float
    power: float
    state: str = "ATTACK"

def collision_time(a, b, va, vb):
    x, y = b.x-a.x, b.y-a.y
    u, v = vb[0]-va[0], vb[1]-va[1]
    c = x*x+y*y-(a.radius+b.radius)**2
    dot = x*u+y*v
    if c <= 1e-7:
        return 0.0 if dot < -1e-9 else math.inf
    aa = u*u+v*v
    disc = dot*dot-aa*c
    if aa < 1e-12 or dot >= 0 or disc < 0:
        return math.inf
    return (-dot-math.sqrt(disc))/aa

def exit_time(r, vel, arena):
    limit = arena-r.radius
    c = r.x*r.x+r.y*r.y-limit*limit
    if limit < 0 or c > 1e-7:
        return 0.0
    aa = vel[0]**2+vel[1]**2
    if aa < 1e-12:
        return math.inf
    dot = r.x*vel[0]+r.y*vel[1]
    return max(0.0, (-dot+math.sqrt(max(0.0, dot*dot-aa*c)))/aa)

def advance(r, v, t):
    return replace(r, x=r.x+v[0]*t, y=r.y+v[1]*t)

def move(a, b, ca, cb, arena, cfg=Config(), trace=False):
    velocities = []
    for r, c in ((a, ca), (b, cb)):
        if not math.isfinite(c.angle) or not math.isfinite(c.power) or not 0 <= c.power <= 1:
            raise ValueError("Action requires finite angle and power in [0,1]")
        d = cfg.distance*r.drive*c.power
        velocities.append((math.cos(c.angle)*d, math.sin(c.angle)*d))
    va, vb = velocities
    frames = [(0.0, a, b)] if trace else []
    contact = collision_time(a, b, va, vb)
    t = 0.0
    collided = False
    for end in ([contact, 1.0] if contact <= 1.0 else [1.0]):
        duration = end-t
        ea, eb = exit_time(a, va, arena), exit_time(b, vb, arena)
        first = min(ea, eb)
        if first <= duration + 1e-10:
            a, b = advance(a, va, first), advance(b, vb, first)
            winner = -1 if abs(ea-eb) < 1e-8 else (1 if ea < eb else 0)
            if trace: frames.append((t+first, a, b))
            return a, b, winner, collided, frames
        a, b = advance(a, va, duration), advance(b, vb, duration)
        t = end
        if trace: frames.append((t, a, b))
        if end == contact and not collided:
            nx, ny = norm(b.x-a.x, b.y-a.y)
            rel = (vb[0]-va[0])*nx+(vb[1]-va[1])*ny
            j = -(1+cfg.restitution)*min(0.0, rel)/(1/a.mass+1/b.mass)
            va = (va[0]-j/a.mass*nx, va[1]-j/a.mass*ny)
            vb = (vb[0]+j/b.mass*nx, vb[1]+j/b.mass*ny)
            collided = True
    return a, b, None, collided, frames

def effective_push(before, after, command, elapsed, cfg):
    free=advance(before,(math.cos(command.angle)*cfg.distance*before.drive*command.power,
                         math.sin(command.angle)*cfg.distance*before.drive*command.power),elapsed)
    actual_gain=math.hypot(after.x,after.y)-math.hypot(before.x,before.y)
    causal_gain=math.hypot(after.x,after.y)-math.hypot(free.x,free.y)
    return max(0.0,min(actual_gain,causal_gain))

def initial(seed, cfg=Config(), masses=(1.0, 1.0)):
    rng = random.Random(seed)
    angle = rng.uniform(-math.pi, math.pi)
    sep = rng.uniform(85, 125)
    cx, cy = rng.uniform(-35,35), rng.uniform(-35,35)
    skew = rng.uniform(-30,30)
    a = Robot(cx-sep*math.cos(angle), cy-sep*math.sin(angle), masses[0])
    b = Robot(cx+sep*math.cos(angle)-skew*math.sin(angle),
              cy+sep*math.sin(angle)+skew*math.cos(angle), masses[1])
    return a,b

def match(pa, pb, seed=0, cfg=Config(), masses=(1.0,1.0), record=False, swap=False):
    a,b = initial(seed,cfg,masses)
    if swap: a,b=b,a
    pa.reset(); pb.reset()
    arena = cfg.arena
    history = []
    contacts = 0
    pushes=[0.0,0.0]; tiny_contacts=0; stalled_turns=0
    for turn in range(1,cfg.rounds+1):
        ca,cb = pa.act(a,b,arena,cfg),pb.act(b,a,arena,cfg)
        before_a,before_b=a,b
        a,b,winner,hit,frames = move(a,b,ca,cb,arena,cfg,True)
        travelled=math.hypot(a.x-before_a.x,a.y-before_a.y)+math.hypot(b.x-before_b.x,b.y-before_b.y)
        stalled_turns+=travelled<20
        if hit:
            gain_a=effective_push(before_b,b,cb,frames[-1][0],cfg)
            gain_b=effective_push(before_a,a,ca,frames[-1][0],cfg)
            pushes[0]+=gain_a;pushes[1]+=gain_b
            tiny_contacts+=max(gain_a,gain_b)<15
        contacts += hit
        next_arena = arena
        reason = "exit" if winner is not None else "playing"
        if winner is None and turn == cfg.rounds:
            winner,reason = -1,"round_limit"
        elif winner is None:
            next_arena = arena*cfg.shrink
            out_a = math.hypot(a.x,a.y)+a.radius > next_arena
            out_b = math.hypot(b.x,b.y)+b.radius > next_arena
            if out_a or out_b:
                winner = -1 if out_a and out_b else (1 if out_a else 0)
                reason = "shrink"
        if record:
            history.append(dict(turn=turn,arena=arena,next_arena=next_arena,
                                actions=[asdict(ca),asdict(cb)],
                                frames=[(s,asdict(x),asdict(y)) for s,x,y in frames],
                                collision=hit,winner=winner,reason=reason))
        arena = next_arena
        if winner is not None:
            return dict(winner=winner,reason=reason,turns=turn,contacts=contacts,
                        seed=seed,swap=swap,history=history,push_cm=pushes,tiny_contacts=tiny_contacts,stalled_turns=stalled_turns)

# controllers.py
import json
import math
import random
from pathlib import Path

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
        self.last_position=None
        self.flank_side=1

    def act(self, me, enemy, arena, cfg=Config()):
        p=self.p
        if self.rng is None:
            self.rng=random.Random(round(me.x*1000)*1000003+round(me.y*1000))
            self.flank_side=self.rng.choice((-1,1))
        d=math.hypot(enemy.x-me.x,enemy.y-me.y)
        edge=(math.hypot(me.x,me.y)+me.radius)/arena
        close=d/arena
        self.stalled = self.stalled+1 if self.last_distance is not None and abs(d-self.last_distance)<8 else 0
        if self.last_position is not None and math.hypot(me.x-self.last_position[0],me.y-self.last_position[1])<10:
            self.stalled=max(self.stalled,1)
        self.last_position=(me.x,me.y)
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
            danger=ramp(edge,p["edge"],p["width"])
            near=1-ramp(close,p["close"],p["width"])
            stuck=min(1.0,self.stalled/2)
            near=max(near,stuck)
            weights=[danger,(1-danger)*near*(1-stuck),
                     (1-danger)*(1-near),(1-danger)*near*stuck]
            outputs=[(2.5,.5,0),(.55,1.7,0),(.8,1,0),(.8,1.2,p["flank"])]
            total=sum(weights)
            safety=p["safety"]*sum(w*v[0] for w,v in zip(weights,outputs))/total
            pressure=p["pressure"]*sum(w*v[1] for w,v in zip(weights,outputs))/total
            flank=sum(w*v[2] for w,v in zip(weights,outputs))/total
            approach=p["approach"]
            self.state="FUZZY_"+["RECOVER","PUSH","APPROACH","FLANK"][max(range(4),key=weights.__getitem__)]
        heading=math.atan2(enemy.y-me.y,enemy.x-me.x)
        center=math.atan2(-me.y,-me.x)
        predicted=Action(heading+math.pi,1)
        angles=[heading,heading+p["flank"],heading-p["flank"],
                heading+1.15,heading-1.15,center]
        contact_angle=math.asin(min(.95,(me.radius+enemy.radius)/max(d,1e-6)))*.65
        angles.extend([heading+contact_angle,heading-contact_angle])
        minimum=cfg.min_distance/(cfg.distance*me.drive)
        if not 0<=minimum<=1:raise ValueError("Invalid minimum travel for this robot")
        candidates=[Action(a,power,self.state) for a in angles for power in sorted({max(minimum,x) for x in (p["power"],.5,.15,1.0)})]
        best=None
        for c in candidates:
            a,b,w,hit,preview=move(me,enemy,c,predicted,arena,cfg,True)
            if w is None:
                out_a=math.hypot(a.x,a.y)+a.radius>arena*cfg.shrink
                out_b=math.hypot(b.x,b.y)+b.radius>arena*cfg.shrink
                if out_a or out_b: w=-1 if out_a and out_b else (1 if out_a else 0)
            ra=math.hypot(a.x,a.y)/arena
            rb=math.hypot(b.x,b.y)/arena
            gap=math.hypot(a.x-b.x,a.y-b.y)/arena
            outcome=100 if w==0 else (-100 if w==1 else (-25 if w==-1 else 0))
            risk=max(0,(math.hypot(a.x,a.y)+a.radius)/(arena*cfg.shrink)-.85)
            offset=abs(math.atan2(math.sin(c.angle-heading),math.cos(c.angle-heading)))
            value=(outcome+pressure*rb-safety*(ra*.25+5*risk)
                   -approach*gap+4*flank*min(offset,1.15)+.03*c.power)
            if hit:
                push=effective_push(enemy,b,predicted,preview[-1][0],cfg)
                value+=p["pressure"]*min(2.0,push/cfg.distance)*.5
            if self.stalled:
                signed=math.atan2(math.sin(c.angle-heading),math.cos(c.angle-heading))
                value+=self.rng.uniform(-2,2)*min(self.stalled,2)
                value+=.3*flank*self.flank_side*math.sin(signed)
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

# train.py
import argparse
import json
import random
from pathlib import Path


def opponents(champions):
    return [Baseline("charge"),Baseline("center"),Controller("fsm"),Controller("fuzzy")]+[
        Controller(k,p) for k,p in champions]

def score(kind,params,seeds,pool):
    total=0.0; n=0
    for seed in seeds:
        for opponent in pool:
            for swap in (False,True):
                r=match(Controller(kind,params),opponent,seed,swap=swap)
                reward=10 if r["winner"]==0 else (-10 if r["winner"]==1 else -2)
                if r["winner"]==1 and r["reason"]=="exit" and r["contacts"]==0: reward-=2
                cfg=Config()
                reward+=cfg.push_reward*min(1.0,r["push_cm"][0]/(2*cfg.distance))
                reward-=cfg.stall_penalty*r["stalled_turns"]/cfg.rounds
                total+=reward; n+=1
    return total/n

def train(generations=12,population=16,seed=20261105):
    if generations<1 or population<4: raise ValueError("generations >=1, population >=4")
    rng=random.Random(seed)
    folder=ROOT/"models"; folder.mkdir(exist_ok=True)
    champions=[]; logs=[]
    reference=opponents([])
    control_seeds=list(range(2100000,2100008))
    for kind in ("fsm","fuzzy"):
        initial_params=load(folder/f"{kind}.json").p if (folder/f"{kind}.json").exists() else dict(DEFAULT)
        pop=[dict(initial_params)]+[{k:rng.uniform(*bounds) for k,bounds in BOUNDS.items()}
                             for _ in range(population-1)]
        archive=[]
        incumbent=dict(initial_params)
        control_best=score(kind,incumbent,control_seeds,reference)
        for gen in range(generations):
            seeds=[rng.randrange(1,1000000) for _ in range(3)]
            pool=opponents(champions[-3:])
            ranked=sorted([(score(kind,p,seeds,pool),p) for p in pop],key=lambda x:x[0],reverse=True)
            archive.append(ranked[0][1].copy())
            row=dict(kind=kind,generation=gen+1,best=ranked[0][0],
                     mean=sum(x[0] for x in ranked)/population,seeds=seeds)
            control=score(kind,ranked[0][1],control_seeds,reference)
            if control>control_best:
                control_best=control; incumbent=ranked[0][1].copy()
            row.update(control_candidate=control,control_best=control_best)
            logs.append(row)
            (folder/"training_log.json").write_text(json.dumps(logs,indent=2),encoding="utf-8")
            print(f"{kind:5} generation {gen+1:2}/{generations} fitness={row['best']:.3f}",flush=True)
            if gen%3==2: champions.append((kind,ranked[0][1].copy()))
            pop=[ranked[0][1].copy(),ranked[1][1].copy()]
            def parent(): return max(rng.sample(ranked,min(3,len(ranked))),key=lambda x:x[0])[1]
            while len(pop)<population:
                a,b=parent(),parent(); child={}
                for k,(lo,hi) in BOUNDS.items():
                    value=a[k] if rng.random()<.5 else b[k]
                    if rng.random()<.5: value+=rng.gauss(0,(hi-lo)*.13)
                    child[k]=max(lo,min(hi,value))
                pop.append(child)
        validation=list(range(2000000,2000012))
        pool=opponents(champions[-3:])
        candidates=[dict(initial_params),dict(DEFAULT),incumbent]+archive
        scores=[score(kind,p,validation,pool) for p in candidates]
        idx=max(range(len(scores)),key=scores.__getitem__)
        chosen=candidates[idx]
        metadata=dict(training_seed=seed,generations=generations,population=population,
                      validation_fitness=scores[idx],default_validation_fitness=scores[1],initial_validation_fitness=scores[0],
                      selected_candidate=idx,validation_seeds=validation,
                      algorithm="elitist EA, tournament/crossover/Gaussian mutation")
        Controller(kind,chosen).save(folder/f"{kind}.json",metadata)
        champions.append((kind,chosen))
    (folder/"training_log.json").write_text(json.dumps(logs,indent=2),encoding="utf-8")

def benchmark(n=100,seed_start=5000000):
    from collections import Counter
    entries={"fsm":lambda:load(ROOT/"models/fsm.json"),
             "fuzzy":lambda:load(ROOT/"models/fuzzy.json"),
             "default_fsm":lambda:Controller("fsm"),
             "default_fuzzy":lambda:Controller("fuzzy"),
             "charge":lambda:Baseline("charge"),"center":lambda:Baseline("center")}
    pairs=[("fsm","fsm"),("fsm","fuzzy"),("fuzzy","fuzzy")]+[
        (k,other) for k in ("fsm","fuzzy") for other in ("charge","center","default_fsm","default_fuzzy")]
    rows=[]
    for ka,kb in pairs:
        count=Counter(); turns=0; contacts=0; push_total=0; tiny=0; stalls=0
        for seed in range(seed_start,seed_start+n):
            for swap in (False,True):
                r=match(entries[ka](),entries[kb](),seed,swap=swap)
                count[{0:"wins_a",1:"wins_b",-1:"draws"}[r["winner"]]]+=1
                count[r["reason"]]+=1
                turns+=r["turns"]; contacts+=r["contacts"]
                push_total+=sum(r["push_cm"]);tiny+=r["tiny_contacts"];stalls+=r["stalled_turns"]
        row=dict(a=ka,b=kb,games=2*n,**{k:count[k] for k in (
            "wins_a","wins_b","draws","exit","shrink","round_limit")},
                 mean_turns=turns/(2*n),mean_contacts=contacts/(2*n),mean_push_cm=push_total/(2*n),mean_tiny_contacts=tiny/(2*n),mean_stalled_turns=stalls/(2*n))
        rows.append(row); print(row,flush=True)
    result=dict(seed_start=seed_start,starts=n,paired_side_swaps=True,results=rows)
    (ROOT/"benchmark.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result



# demo.py
"""Animated match viewer: python demo.py --a fsm --b fuzzy --seed 42"""
import argparse
import json
import math
from pathlib import Path


def resolve(name):
    return load(ROOT/"models"/(name+".json") if name in ("fsm","fuzzy") else Path(name))

def animate(result,a_name,b_name,save=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from matplotlib.patches import Circle
    fig,ax=plt.subplots(figsize=(9,9),facecolor="#101827")
    ax.set_facecolor("#101827"); ax.set_aspect("equal")
    limit=Config().arena*1.12;ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit);ax.axis("off")
    arena=Circle((0,0),900,fc="#202e43",ec="#dde8f2",lw=2);ax.add_patch(arena)
    robots=[Circle((0,0),50,color=c) for c in ("#ff685d","#38c6ef")]
    for robot in robots:ax.add_patch(robot)
    trails=[ax.plot([],[],color=c,alpha=.6,lw=2)[0] for c in ("#ff685d","#38c6ef")]
    labels=[ax.text(0,0,x,ha="center",va="center",weight="bold") for x in ("A","B")]
    title=ax.text(.5,1.01,f"A: {a_name.upper()}   vs   B: {b_name.upper()}",
                  transform=ax.transAxes,ha="center",color="white",fontsize=16)
    info=ax.text(.02,.98,"",transform=ax.transAxes,va="top",color="white",fontsize=10)
    outcome={0:"WYGRYWA A",1:"WYGRYWA B",-1:"REMIS"}[result["winner"]]
    reasons={"exit":"przekroczenie granicy", "shrink":"zmniejszenie areny", "round_limit":"limit 10 rund"}
    frames=[]
    for h in result["history"]:
        events=h["frames"]
        for frame in range(25):
            time=frame/24
            s=(1-math.exp(-Config().drag*time))/(1-math.exp(-Config().drag))
            s*=events[-1][0]
            j=next((j for j in range(1,len(events)) if events[j][0]>=s),len(events)-1)
            prev,nxt=events[max(0,j-1)],events[j]
            t=(s-prev[0])/max(1e-12,nxt[0]-prev[0])
            pos=[(prev[k]["x"]+(nxt[k]["x"]-prev[k]["x"])*t,
                  prev[k]["y"]+(nxt[k]["y"]-prev[k]["y"])*t) for k in (1,2)]
            frames.append((h,pos,h["arena"],False))
        for _ in range(8):frames.append((h,pos,h["next_arena"],True))
    frames.extend([frames[-1]]*35)
    paths=[[],[]]
    for _,pos,_,_ in frames:
        for k in range(2):paths[k].append(pos[k])
    def update(i):
        h,pos,radius,paused=frames[i]
        arena.set_radius(radius)
        for k in range(2):
            robots[k].center=pos[k];labels[k].set_position(pos[k])
            trail=paths[k][max(0,i-120):i+1]
            trails[k].set_data([p[0] for p in trail],[p[1] for p in trail])
        ca,cb=h["actions"]
        info.set_text(f"Runda {h['turn']}/10 | promien {radius:.1f}\n"
                      f"A: {ca['state']}  moc {ca['power']:.2f}\n"
                      f"B: {cb['state']}  moc {cb['power']:.2f}\n"
                      f"Kolizja: {'TAK' if h['collision'] else 'NIE'}")
        if h["winner"] is not None and paused:
            title.set_text(f"{outcome} - {reasons[result['reason']]}")
        else:title.set_text(f"A: {a_name.upper()}   vs   B: {b_name.upper()}")
        return robots+labels+trails+[arena,info,title]
    anim=FuncAnimation(fig,update,frames=len(frames),interval=40,repeat=False)
    fig._animation=anim
    if save:anim.save(save,writer=PillowWriter(fps=25));plt.close(fig)
    else:plt.show()


EMBEDDED_MODELS = {'fsm': {'schema': 1, 'kind': 'fsm', 'parameters': {'edge': 0.597888681377301, 'close': 0.22414751077340817, 'flank': 0.18, 'power': 0.7884069014596851, 'safety': 4.0, 'pressure': 1.3458374844556682, 'approach': 1.8151041029046617, 'width': 0.06211971335511562, 'hysteresis': 0.06341054738308549}, 'metadata': {'training_seed': 20261114, 'generations': 4, 'population': 10, 'validation_fitness': 1.2589274583759908, 'default_validation_fitness': -0.894162619988775, 'initial_validation_fitness': -1.8898029192549297, 'selected_candidate': 2, 'validation_seeds': [2000000, 2000001, 2000002, 2000003, 2000004, 2000005, 2000006, 2000007, 2000008, 2000009, 2000010, 2000011], 'algorithm': 'elitist EA, tournament/crossover/Gaussian mutation'}}, 'fuzzy': {'schema': 1, 'kind': 'fuzzy', 'parameters': {'edge': 0.7810082816638335, 'close': 0.3477999195190186, 'flank': 0.4045700412510373, 'power': 0.8292522932491274, 'safety': 4.0, 'pressure': 2.654724733333139, 'approach': 0.3669935494762385, 'width': 0.23353911644269637, 'hysteresis': 0.04300725720367134}, 'metadata': {'training_seed': 20261114, 'generations': 4, 'population': 10, 'validation_fitness': 0.10725215228187057, 'default_validation_fitness': -1.6232782093201228, 'initial_validation_fitness': -2.7129229957298424, 'selected_candidate': 5, 'validation_seeds': [2000000, 2000001, 2000002, 2000003, 2000004, 2000005, 2000006, 2000007, 2000008, 2000009, 2000010, 2000011], 'algorithm': 'elitist EA, tournament/crossover/Gaussian mutation'}}}


ROOT=Path(__file__).resolve().parent/"sumo_push_final"

def prepare_models():
    (ROOT/"models").mkdir(parents=True,exist_ok=True)
    for kind,data in EMBEDDED_MODELS.items():
        path=ROOT/"models"/(kind+".json")
        if not path.exists(): path.write_text(json.dumps(data,indent=2),encoding="utf-8")

def plot_learning():
    import matplotlib.pyplot as plt
    logs=json.loads((ROOT/"models/training_log.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for kind in ("fsm","fuzzy"):
        rows=[r for r in logs if r["kind"]==kind]
        x=[r["generation"] for r in rows]
        axes[0].plot(x,[r["best"] for r in rows],label=kind.upper())
        axes[1].plot(x,[r["control_best"] for r in rows],label=kind.upper())
    axes[0].set_title("Najlepszy osobnik A zmienna pula walk")
    axes[1].set_title("Najlepszy dotychczas A staly zbior kontrolny")
    for ax in axes:
        ax.set_xlabel("Pokolenie");ax.set_ylabel("Fitness");ax.legend();ax.grid(alpha=.3)
    fig.savefig(ROOT/"learning.png",dpi=150);plt.close(fig)

def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--from-scratch",action="store_true",help="Train without embedded champions")
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--generations",type=int,default=16)
    parser.add_argument("--population",type=int,default=18)
    parser.add_argument("--games",type=int,default=100)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--training-seed",type=int,default=20261114)
    parser.add_argument("--a",default="fsm")
    parser.add_argument("--b",default="fuzzy")
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--save",help="Save animation to GIF")
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args();ROOT=args.output.resolve()
    if args.games<1 or args.generations<1 or args.population<4:parser.error("games/generations >=1, population >=4")
    if args.from_scratch and not args.train:parser.error("--from-scratch requires --train")
    prepare_models()
    if args.train:
        from datetime import datetime
        import shutil
        backup=ROOT/("before_training_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        shutil.copytree(ROOT/"models",backup)
        if args.from_scratch:
            for kind in ("fsm","fuzzy"): Controller(kind).save(ROOT/"models"/(kind+".json"))
        train(args.generations,args.population,args.training_seed)
        if not args.headless:plot_learning()
    if args.benchmark or args.train:benchmark(args.games,seed_start=7000000)
    if not args.benchmark:
        result=match(resolve(args.a),resolve(args.b),args.seed,record=True)
        (ROOT/"last_match.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))
        if not args.headless:animate(result,args.a,args.b,args.save)
    print("Wyniki:",ROOT)

if __name__=="__main__":main()
