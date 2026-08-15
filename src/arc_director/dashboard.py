"""Dependency-free live dashboard for Director training.

The trainer already writes an append-only ``train_log.jsonl`` file.  This
module serves that durable log beside a small static page, so observing a run
does not add a web-framework dependency or another metrics format that can
drift away from the source of truth.
"""

from __future__ import annotations

import functools
import html
import http.server
import json
import os
import socketserver
import threading
from pathlib import Path
from typing import Any, Optional

__all__ = ["DashboardServer"]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class DashboardServer:
    """Write and optionally serve the dashboard for one run directory."""

    def __init__(
        self,
        run_dir: str | Path,
        title: str = "ARC Director",
        *,
        port: Optional[int] = 8321,
        phase: str = "Training",
        phase_index: int = 1,
        phase_total: int = 1,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "dashboard.json"
        self._state: dict[str, Any] = {}
        self._server: Optional[_ThreadingServer] = None
        self.url: Optional[str] = None
        self.update(
            status="starting",
            phase=phase,
            phase_index=int(phase_index),
            phase_total=int(phase_total),
        )
        safe_title = html.escape(title, quote=True)
        (self.run_dir / "index.html").write_text(
            _PAGE.replace("{{RUN}}", safe_title), encoding="utf-8"
        )
        if port is not None:
            self._serve(int(port))

    def _serve(self, port: int) -> None:
        handler = functools.partial(_QuietHandler, directory=str(self.run_dir))
        try:
            self._server = _ThreadingServer(("127.0.0.1", port), handler)
        except OSError as error:
            print(
                f"[dashboard] port {port} unavailable ({error}); training will continue",
                flush=True,
            )
            return
        actual_port = int(self._server.server_address[1])
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{actual_port}/"
        print(f"[dashboard] {self.url}", flush=True)

    def update(self, **values: Any) -> dict[str, Any]:
        """Atomically publish launcher/trainer state for the polling page."""
        self._state.update(values)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, self.state_path)
        return dict(self._state)

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> "DashboardServer":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{RUN}} · live training</title>
<style>
  :root { color-scheme:dark; --bg:#090c12; --panel:#121824; --line:#273248;
    --text:#edf4ff; --dim:#91a0b7; --blue:#58a6ff; --green:#43d18b;
    --amber:#ffbd66; --purple:#ba9cff; --red:#ff6b7a; }
  * { box-sizing:border-box; }
  body { margin:0; color:var(--text); background:
    radial-gradient(circle at 20% -10%,#182945 0,transparent 35%),var(--bg);
    font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex;
    gap:16px; align-items:baseline; flex-wrap:wrap; }
  h1 { margin:0; font-size:17px; font-weight:700; letter-spacing:.2px; }
  #status { color:var(--dim); font-size:12px; font-variant-numeric:tabular-nums; }
  main { padding:16px 24px 34px; display:grid; gap:12px;
    grid-template-columns:repeat(auto-fit,minmax(285px,1fr)); }
  .card { min-width:0; padding:14px 16px; border:1px solid var(--line);
    border-radius:10px; background:linear-gradient(145deg,#151c29,#101620); }
  .wide { grid-column:1/-1; }
  h2 { margin:0 0 4px; color:var(--dim); font-size:10px; letter-spacing:.9px;
    text-transform:uppercase; font-weight:700; }
  .value { font-size:28px; font-weight:720; letter-spacing:-.6px;
    font-variant-numeric:tabular-nums; }
  .value small { color:var(--dim); font-size:12px; font-weight:500; letter-spacing:0; }
  .pipeline { display:grid; gap:8px; grid-template-columns:minmax(180px,.7fr) 2fr; }
  .pipeline strong { font-size:18px; }
  .pipeline p { color:var(--dim); margin:0; }
  .bar { height:7px; background:#0a0f17; border-radius:8px; overflow:hidden; margin-top:10px; }
  .bar > i { display:block; height:100%; width:0; background:linear-gradient(90deg,var(--blue),var(--green)); }
  canvas { display:block; width:100%; height:88px; margin-top:7px; }
  .legend { color:var(--dim); font-size:11px; margin-top:3px; }
  .green { color:var(--green); } .amber { color:var(--amber); }
  @media(max-width:620px){ main,header{padding-left:14px;padding-right:14px}.pipeline{grid-template-columns:1fr} }
</style>
</head>
<body>
<header><h1>{{RUN}}</h1><span id="status">waiting for the first update…</span></header>
<main>
  <section class="card wide pipeline">
    <div><h2>Training pipeline</h2><strong id="phase">Starting</strong><div id="phaseCount" class="legend"></div><div id="contract" class="legend">Waiting for hierarchy contract.</div></div>
    <div><h2>Current curriculum rung</h2><div class="value" id="stage">—</div>
      <p id="stageDetail">Generated DSL programs come first; ARC tasks follow after the policy earns promotion.</p>
      <div class="bar"><i id="stageBar"></i></div></div>
  </section>
  <section class="card"><h2>Held-out generalization</h2><div class="value green" id="gen">—</div><canvas id="genChart"></canvas><div class="legend">The promotion metric; not scored during the episode.</div></section>
  <section class="card"><h2>Demonstration solve rate</h2><div class="value" id="solve">—</div><canvas id="solveChart"></canvas></section>
  <section class="card"><h2>Combined ARC evaluation</h2><div class="value" id="eval">—</div><canvas id="evalChart"></canvas><div class="legend" id="evalDetail">Appears on held-out ARC dev tasks.</div></section>
  <section class="card"><h2>ARC-AGI-1 exact @ 2</h2><div class="value green" id="arc1">—</div><canvas id="arc1Chart"></canvas><div class="legend" id="arc1Detail">Waiting for held-out ARC-1 evaluation.</div></section>
  <section class="card"><h2>ARC-AGI-2 exact @ 2</h2><div class="value green" id="arc2">—</div><canvas id="arc2Chart"></canvas><div class="legend" id="arc2Detail">Waiting for held-out ARC-2 evaluation.</div></section>
  <section class="card"><h2>Episode return</h2><div class="value" id="ret">—</div><canvas id="retChart"></canvas></section>
  <section class="card"><h2>Worker entropy</h2><div class="value" id="entropy">—</div><canvas id="entropyChart"></canvas><div class="legend">A sudden fall can signal policy collapse.</div></section>
  <section class="card"><h2>Worker epiplexity duel</h2><div class="value" id="workerDuel">—</div><canvas id="workerDuelChart"></canvas><div class="legend" id="workerDuelDetail">Waiting for a crossed duel.</div></section>
  <section class="card"><h2>Director epiplexity duel</h2><div class="value" id="directorDuel">—</div><canvas id="directorDuelChart"></canvas><div class="legend" id="directorDuelDetail">Waiting for a crossed duel.</div></section>
  <section class="card"><h2>Top operator share</h2><div class="value" id="top">—</div><canvas id="topChart"></canvas><div class="legend" id="topName">No action data yet.</div></section>
  <section class="card"><h2>Goal progress reward</h2><div class="value" id="goal">—</div><canvas id="goalChart"></canvas></section>
  <section class="card"><h2>Total loss</h2><div class="value" id="loss">—</div><canvas id="lossChart"></canvas></section>
  <section class="card wide"><h2>Throughput and generated programs</h2><div class="value" id="throughput">—</div><div class="legend" id="generated">Waiting for tasks.</div></section>
</main>
<script>
const $=id=>document.getElementById(id);
const pct=v=>Number.isFinite(Number(v))?(100*Number(v)).toFixed(1)+'%':'—';
const num=(v,d=3)=>Number.isFinite(Number(v))?Number(v).toFixed(d):'—';
async function textFile(name){try{const r=await fetch(name+'?t='+Date.now());return r.ok?await r.text():''}catch(_){return ''}}
async function jsonFile(name){try{const r=await fetch(name+'?t='+Date.now());return r.ok?await r.json():{}}catch(_){return {}}}
function rows(text){return text.trim().split(/\r?\n/).filter(Boolean).map(line=>{try{return JSON.parse(line)}catch(_){return null}}).filter(Boolean)}
function values(data,key){return data.map(r=>Number(r[key])).filter(Number.isFinite)}
function chart(id,data,key,color,zero=false,one=false){
  const c=$(id),rect=c.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
  c.width=Math.max(1,Math.floor(rect.width*ratio));c.height=Math.max(1,Math.floor(rect.height*ratio));
  const x=c.getContext('2d');x.scale(ratio,ratio);const w=rect.width,h=rect.height,p=10;
  x.strokeStyle='#29344a';x.lineWidth=1;x.beginPath();x.moveTo(p,h-p);x.lineTo(w-p,h-p);x.stroke();
  const vs=values(data,key);if(!vs.length)return;let lo=zero?0:Math.min(...vs),hi=one?1:Math.max(...vs);
  if(hi===lo){lo-=.5;hi+=.5}x.strokeStyle=color;x.lineWidth=2;x.beginPath();
  vs.forEach((v,i)=>{const xx=p+(w-2*p)*(vs.length===1?1:i/(vs.length-1));const yy=h-p-(h-2*p)*(v-lo)/(hi-lo);i?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke();
}
async function refresh(){
  const [raw,meta]=await Promise.all([textFile('train_log.jsonl'),jsonFile('dashboard.json')]);
  const all=rows(raw),train=all.filter(r=>r.event!=='eval'),evals=all.filter(r=>r.event==='eval');
  $('phase').textContent=meta.phase||'Training';$('phaseCount').textContent=(meta.phase_total||1)>1?'Phase '+meta.phase_index+' of '+meta.phase_total:'';
  if(!train.length){$('status').textContent=meta.status||'waiting for the first update…';return}
  const r=train[train.length-1],e=evals[evals.length-1];
  $('contract').textContent=r['hierarchy/director_proper']?'Director proper · one latent every '+Number(r['hierarchy/director_every']||1)+' action · worker task-reward weight '+num(r['hierarchy/worker_task_weight'],1):'Hybrid hierarchy';
  $('status').textContent='update '+r.update+' · '+Number(r.env_steps||0).toLocaleString()+' steps · '+num(r.sps,0)+' steps/s';
  $('stage').textContent=r['env/stage']||'—';const rate=Number(r['env/stage_rate']||0),target=Number(r['env/stage_target']||0),episodes=Number(r['env/stage_episodes']||0),minimum=Number(r['env/stage_min_episodes']||0);
  $('stageDetail').textContent='rung '+(Number(r['env/stage_index']||0)+1)+' of '+Number(r['env/stage_total']||1)+' · '+episodes.toLocaleString()+' / '+minimum.toLocaleString()+' episodes · generalization '+pct(rate)+' / '+pct(target);
  $('stageBar').style.width=Math.max(0,Math.min(100,target?rate*100/target:0))+'%';
  $('gen').textContent=pct(r.generalize_rate);$('solve').textContent=pct(r.solve_rate);$('ret').textContent=num(r.mean_return,2);
  $('entropy').textContent=num(r['policy/worker_entropy'],3);$('top').textContent=pct(r['actions/top_op_share']);
  $('workerDuel').textContent=r['epiplexity/worker_winner']||'off';$('directorDuel').textContent=r['epiplexity/director_winner']||'off';
  $('workerDuelDetail').textContent='AUC explore '+num(r['epiplexity/worker_explore_auc'],5)+' · exploit '+num(r['epiplexity/worker_exploit_auc'],5)+' · explore win rate '+pct(r['epiplexity/worker_explore_win_rate']);
  $('directorDuelDetail').textContent='AUC explore '+num(r['epiplexity/director_explore_auc'],5)+' · exploit '+num(r['epiplexity/director_exploit_auc'],5)+' · explore win rate '+pct(r['epiplexity/director_explore_win_rate']);
  $('topName').textContent=(r['actions/top_op']||'—')+' · '+(Number(r['actions/distinct_ops']||0))+' distinct operators in the rollout';
  $('goal').textContent=num(r['reward/goal_mean'],4);$('loss').textContent=num(r['loss/total'],4);
  $('throughput').innerHTML=num(r.sps,0)+' <small>environment steps / second</small>';
  $('generated').textContent=Number.isFinite(Number(r['env/generated']))?Number(r['env/generated']).toLocaleString()+' self-generated tasks · rejection rate '+pct(r['env/reject_rate'])+' · longest rejection streak '+Number(r['env/max_rejection_streak']||0).toLocaleString()+' · '+Number(r.episodes||0).toLocaleString()+' completed episodes':Number(r['env/pool']||0).toLocaleString()+' ARC tasks in this rung · '+Number(r['env/tasks_solved']||0).toLocaleString()+' solved during training · '+Number(r.episodes||0).toLocaleString()+' completed episodes';
  if(e){
    const attempts=e['eval/n_attempts']||'N';$('eval').textContent=pct(e['eval/exact_at_2']);$('evalDetail').textContent='demo fit '+pct(e['eval/demo_fit_rate'])+' · pass@'+attempts+' '+pct(e['eval/pass_at_n']);
    $('arc1').textContent=pct(e['eval/arc1/exact_at_2']);$('arc1Detail').textContent=Number(e['eval/arc1/tasks']||0)+' tasks · demo fit '+pct(e['eval/arc1/demo_fit_rate'])+' · pass@'+attempts+' '+pct(e['eval/arc1/pass_at_n']);
    $('arc2').textContent=pct(e['eval/arc2/exact_at_2']);$('arc2Detail').textContent=Number(e['eval/arc2/tasks']||0)+' tasks · demo fit '+pct(e['eval/arc2/demo_fit_rate'])+' · pass@'+attempts+' '+pct(e['eval/arc2/pass_at_n']);
  }
  chart('genChart',train,'generalize_rate','#43d18b',true,true);chart('solveChart',train,'solve_rate','#58a6ff',true,true);
  chart('evalChart',evals,'eval/exact_at_2','#43d18b',true,true);chart('arc1Chart',evals,'eval/arc1/exact_at_2','#58a6ff',true,true);chart('arc2Chart',evals,'eval/arc2/exact_at_2','#43d18b',true,true);chart('retChart',train,'mean_return','#ffbd66');
  chart('entropyChart',train,'policy/worker_entropy','#ba9cff',true);chart('topChart',train,'actions/top_op_share','#ff6b7a',true,true);
  chart('workerDuelChart',train,'epiplexity/worker_explore_win_rate','#58a6ff',true,true);chart('directorDuelChart',train,'epiplexity/director_explore_win_rate','#ba9cff',true,true);
  chart('goalChart',train,'reward/goal_mean','#58a6ff');chart('lossChart',train,'loss/total','#ffbd66');
}
refresh();setInterval(refresh,2000);window.addEventListener('resize',refresh);
</script>
</body>
</html>
"""
