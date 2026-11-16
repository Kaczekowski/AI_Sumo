
from dataclasses import asdict
import math
from controllers import load
from physics import Robot,Config

class Agent:
    def __init__(self,model_path): self.controller=load(model_path)
    def reset(self): self.controller.reset()
    def decide(self,observation):
        
        me=Robot(**observation["me"]);enemy=Robot(**observation["enemy"])
        cfg=Config(**observation.get("config",{}))
        c=self.controller.act(me,enemy,observation["arena_radius"],cfg)
        return dict(direction_x=math.cos(c.angle),direction_y=math.sin(c.angle),
                    yaw_degrees=math.degrees(c.angle),power=c.power,state=c.state,
                    planned_distance_cm=cfg.distance*me.drive*c.power)
