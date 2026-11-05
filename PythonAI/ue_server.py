"""Optional local HTTP bridge for UE runtime. Run: python ue_server.py

POST /start {"a":"fsm","b":"fuzzy"} -> {"session":"..."}
POST /decide {"session":"...","a":{robot},"b":{robot},"arena_radius":300}
-> {"a":{command},"b":{command}}
POST /end {"session":"..."} -> {"ok":true}
Only loopback, fixed model names, single-threaded ordered decision calls.
"""
import argparse
import json
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
from ue_bridge import Agent

ROOT=Path(__file__).resolve().parent
sessions={}

def dispatch(path,data):
    if path=="/start":
        if len(sessions)>=64: raise ValueError("End old sessions first (limit 64)")
        names=[data.get(k,default) for k,default in (("a","fsm"),("b","fuzzy"))]
        if any(x not in ("fsm","fuzzy") for x in names): raise ValueError("Unknown model")
        token=uuid.uuid4().hex
        sessions[token]=[Agent(ROOT/"models"/(x+".json")) for x in names]
        return dict(session=token)
    agents=sessions[data["session"]]
    if path=="/end":
        del sessions[data["session"]]
        return dict(ok=True)
    if path!="/decide": raise ValueError("Unknown endpoint")
    common=dict(arena_radius=data["arena_radius"],config=data.get("config",{}))
    return {k:agent.decide(dict(me=data[k],enemy=data[other],**common))
            for k,other,agent in (("a","b",agents[0]),("b","a",agents[1]))}

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<=65536: raise ValueError("Invalid body length")
            result=dispatch(self.path,json.loads(self.rfile.read(length)))
            body=json.dumps(result,allow_nan=False).encode()
            self.send_response(200)
        except (ValueError,KeyError,TypeError) as exc:
            body=json.dumps(dict(error=str(exc))).encode();self.send_response(400)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers();self.wfile.write(body)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    print(f"Local UE bridge: http://127.0.0.1:{args.port} (Ctrl+C to stop)",flush=True)
    try:HTTPServer(("127.0.0.1",args.port),Handler).serve_forever()
    except KeyboardInterrupt:pass
