"""Deterministic event-driven 2D physics; no AI or GUI dependencies."""
from dataclasses import dataclass, replace, asdict
import math
import random

def norm(x, y):
    d = math.hypot(x, y)
    return (x/d, y/d) if d > 1e-12 else (1.0, 0.0)

@dataclass(frozen=True)
class Config:
    arena: float = 300.0
    shrink: float = 0.86
    rounds: int = 10
    distance: float = 155.0
    restitution: float = 0.25
    # Uniform drag: v(t)=v0 exp(-drag*t); distance is the integrated full-turn travel.
    drag: float = 0.7

@dataclass(frozen=True)
class Robot:
    x: float
    y: float
    mass: float = 1.0
    radius: float = 16.0
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
    """Exact swept contact and first boundary crossing, including the post-impact path.

    Parameter s is integrated drag time normalized to [0,1]. Identical linear
    drag on both bodies preserves straight paths and event ordering in s.
    A collision applies an impulse; commands never change during this turn.
    There can be at most one contact for two free ballistic discs.
    winner: 0=A, 1=B, -1=simultaneous exit, None=no exit.
    """
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

def initial(seed, cfg=Config(), masses=(1.0, 1.0)):
    rng = random.Random(seed)
    # Asymmetric, seeded legal starts; no side is assigned a systematic advantage.
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
    for turn in range(1,cfg.rounds+1):
        # Frozen observations, both decisions before any mutation.
        ca,cb = pa.act(a,b,arena,cfg),pb.act(b,a,arena,cfg)
        a,b,winner,hit,frames = move(a,b,ca,cb,arena,cfg,record)
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
                        seed=seed,swap=swap,history=history)
