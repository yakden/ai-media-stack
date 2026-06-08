"""Interior-3D test console: floor plan -> 3D, three pipelines in one UI.

Modes (tabs):
  • План→3D   — FloorPlanTo3D Mask R-CNN detects walls/doors/windows; the browser
                extrudes them into a navigable 3D scene with Three.js. (server-light)
  • Рендер    — ControlNet + Stable Diffusion (ComfyUI) -> photoreal interior render.
  • Меблировка — ReSpace LLM places real 3D furniture from a text brief.

Each backend is optional and discovered at runtime via /api/status, so the UI
degrades gracefully (a mode shows "бэкенд не установлен" until its service is up).
A free-disk guard refuses heavy backend work when the shared ZFS pool is low —
this host runs a production 1C VM on the same pool (see incident 2026-06-07).
"""
from __future__ import annotations

import asyncio
import io
import os
import subprocess

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

# Backend service URLs (each may or may not be running).
FLOORPLAN_URL = os.environ.get("FLOORPLAN_URL", "http://127.0.0.1:8204")
RENDER_URL = os.environ.get("RENDER_URL", "http://127.0.0.1:8210")    # ComfyUI SD/ControlNet
RESPACE_URL = os.environ.get("RESPACE_URL", "http://127.0.0.1:8211")
MIN_FREE_GB = 12  # refuse heavy backend ops below this (protect the 1C VM's pool)

app = FastAPI(title="interior3d-web")


def _free_gb(path="/"):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


async def _alive(url: str, path: str = "/") -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            r = await c.get(url + path)
            return r.status_code < 500
    except Exception:
        return False


@app.get("/api/status")
async def status():
    # FloorPlanTo3D answers only POST; a GET 405 still proves it's up.
    return {
        "free_gb": round(_free_gb(), 1),
        "min_free_gb": MIN_FREE_GB,
        "backends": {
            "floorplan": await _alive(FLOORPLAN_URL, "/"),
            "render": await _alive(RENDER_URL, "/health"),
            "respace": await _alive(RESPACE_URL, "/health"),
        },
    }


@app.post("/api/floorplan")
async def floorplan(image: UploadFile = File(...)):
    """Forward a plan image to the Mask R-CNN service; return its boxes/classes."""
    raw = await image.read()
    # Normalize to 3-channel RGB PNG: the upstream loader doesn't strip alpha, so
    # RGBA/grayscale/palette plans would otherwise break mold_image (shape mismatch).
    try:
        buf = io.BytesIO()
        Image.open(io.BytesIO(raw)).convert("RGB").save(buf, format="PNG")
        data = buf.getvalue()
    except Exception as exc:
        raise HTTPException(400, f"Не удалось прочитать изображение: {exc}") from exc
    files = {"image": ("plan.png", data, "image/png")}
    try:
        async with httpx.AsyncClient(timeout=180) as c:
            det = await c.post(FLOORPLAN_URL + "/", files=files)
            try:
                ocr = await c.post(FLOORPLAN_URL + "/ocr", files=files)
            except Exception:
                ocr = None
    except Exception as exc:
        raise HTTPException(503, f"FloorPlanTo3D backend недоступен: {exc}") from exc
    if det.status_code != 200:
        raise HTTPException(502, f"detect {det.status_code}: {det.text[:160]}")
    out = det.json()
    # OCR is best-effort: detection still works if it fails.
    if ocr is not None and ocr.status_code == 200:
        o = ocr.json()
        out["rooms"] = o.get("rooms", [])
        out["stairs"] = o.get("stairs", [])
        out["dims"] = o.get("dims", [])
        out["words"] = o.get("words", [])
    return JSONResponse(out)


RENDER_COMPOSE = "/home/deploy/interior-render/docker-compose.yml"
RENDER_EVICT = ["whisper-xtts-server", "avatar-muse"]   # freed to make RAM/VRAM headroom


def _restore_default():
    """Unload the render model and bring the resident 'duty' models back.

    Implements the single-model-at-a-time GPU policy: a render borrows the T4,
    then we swap back to whisper/avatar so STT/TTS isn't left down. Runs after the
    HTTP response (BackgroundTask)."""
    subprocess.run(["sudo", "docker", "stop", "interior-render"], capture_output=True, timeout=120)
    subprocess.run(["sudo", "docker", "start", *RENDER_EVICT], capture_output=True, timeout=120)


async def _render(data: bytes, prompt: str, steps: int, scale: float,
                  bg: BackgroundTasks) -> Response:
    """Shared on-demand SD+ControlNet backend used by both render and furnish."""
    if _free_gb() < MIN_FREE_GB:
        raise HTTPException(507, f"Мало места на диске ({_free_gb():.0f} ГБ). Рендер заблокирован.")
    if not await _alive(RENDER_URL, "/health"):
        subprocess.run(["sudo", "docker", "stop", *RENDER_EVICT], capture_output=True, timeout=120)
        up = subprocess.run(["sudo", "docker", "compose", "-f", RENDER_COMPOSE, "up", "-d"],
                            capture_output=True, text=True, timeout=300)
        if up.returncode != 0:
            raise HTTPException(503, f"не удалось запустить render: {up.stderr[-200:]}")
        for _ in range(90):
            if await _alive(RENDER_URL, "/health"):
                break
            await asyncio.sleep(2)
        else:
            _restore_default()
            raise HTTPException(503, "render backend не поднялся (docker logs interior-render)")
    try:  # first call downloads + loads models -> can take a few minutes
        async with httpx.AsyncClient(timeout=900) as c:
            r = await c.post(RENDER_URL + "/render",
                             data={"prompt": prompt, "steps": steps, "scale": scale},
                             files={"image": ("plan.png", data, "image/png")})
    except Exception as exc:
        bg.add_task(_restore_default)
        raise HTTPException(504, f"render timeout/ошибка: {exc}") from exc
    # swap back to the resident model after the image is sent
    bg.add_task(_restore_default)
    if r.status_code != 200:
        raise HTTPException(502, f"render {r.status_code}: {r.text[:160]}")
    return Response(content=r.content, media_type="image/png")


@app.post("/api/render")
async def render(bg: BackgroundTasks, image: UploadFile = File(...), prompt: str = Form(""),
                 preset: str = Form("mlsd"), steps: int = Form(22), scale: float = Form(1.0)):
    return await _render(await image.read(), prompt, steps, scale, bg)


@app.post("/api/furnish")
async def furnish(bg: BackgroundTasks, image: UploadFile = File(...), brief: str = Form("")):
    # Furnished render via the shared SD+ControlNet backend: the brief drives a
    # furniture-rich prompt over the plan's structure. (True 3D-asset placement
    # via ReSpace is an optional future upgrade — needs an HF token + 3D-FUTURE.)
    prompt = ((brief or "cozy furnished apartment") +
              ", fully furnished, sofa, bed, dining table, wardrobe, rugs, plants, "
              "lamps, decorated rooms, interior design")
    return await _render(await image.read(), prompt, 24, 0.9, bg)


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = """<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Интерьер 3D — тест</title>
<style>
:root{--bg:#0e1116;--card:#171c24;--line:#222a35;--mut:#8b97a7;--acc:#4f8cff}
body{margin:0;background:var(--bg);color:#e6edf3;font:15px system-ui;padding:18px;max-width:1000px;margin:auto}
h1{font-size:19px}
.tabs{display:flex;gap:8px;margin:12px 0;flex-wrap:wrap}
.tab{background:#222a35;border:1px solid var(--line);color:#e6edf3;padding:9px 14px;border-radius:9px;cursor:pointer}
.tab.on{background:var(--acc);border-color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:12px}
input,select,textarea{width:100%;box-sizing:border-box;background:#0b0e13;color:#e6edf3;border:1px solid var(--line);border-radius:9px;padding:10px;font-size:15px;margin:6px 0}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:11px 16px;font-size:15px;cursor:pointer}
button:disabled{opacity:.45}
.mut{color:var(--mut);font-size:13px}.pane{display:none}.pane.on{display:block}
#view{width:100%;height:440px;background:#000;border-radius:12px;border:1px solid var(--line);margin-top:10px}
img.out{max-width:100%;border-radius:12px;border:1px solid var(--line);margin-top:10px}
.pill{display:inline-block;font-size:12px;padding:2px 8px;border-radius:20px;margin-left:6px}
.ok{background:#11331f;color:#2ecc71}.no{background:#3a1212;color:#e74c3c}.od{background:#2c2710;color:#e0a14f}
.legend{font-size:12px;color:var(--mut);margin-top:6px}
.legend b{color:#e6edf3}
</style></head><body>
<h1>🏠 Интерьер 3D — тестовая консоль <span id=disk class=mut></span></h1>
<div class=tabs>
  <div class="tab on" data-p=plan>План → 3D <span id=st-floorplan class="pill no">…</span></div>
  <div class="tab" data-p=render>Фотореализм (рендер) <span id=st-render class="pill no">…</span></div>
  <div class="tab" data-p=furnish>Меблировка (ReSpace) <span id=st-respace class="pill no">…</span></div>
</div>

<div class="pane on" id=pane-plan>
  <div class=card>
    <div class=mut>Поэтажный план (png/jpg). Сервер найдёт стены/двери/окна (Mask R-CNN),
    браузер построит навигируемую 3D-сцену. ЛКМ — вращать, колесо — зум.</div>
    <input id=planFile type=file accept="image/*">
    <button id=planGo>Построить 3D →</button>
    <div id=planStatus class=mut></div>
    <div id=view></div>
    <div class=legend>Легенда: <b style="color:#9aa7b8">стены</b> · <b style="color:#4f8cff">окна</b> · <b style="color:#e0a14f">двери</b> · <b style="color:#e74c3c">лестницы</b> · <b style="color:#fff">подписи комнат</b></div>
    <div id=recog class=mut style="margin-top:8px"></div>
  </div>
</div>

<div class="pane" id=pane-render>
  <div class=card>
    <div class=mut>План + стиль → фотореалистичный рендер (ControlNet + Stable Diffusion).</div>
    <input id=rFile type=file accept="image/*">
    <textarea id=rPrompt placeholder="Стиль: 'scandinavian living room, warm light, photorealistic'"></textarea>
    <select id=rPreset><option value=mlsd>M-LSD (линии стен)</option><option value=depth>Depth</option><option value=seg>Segmentation</option></select>
    <button id=rGo>Сгенерировать рендер →</button>
    <div id=rStatus class=mut></div>
    <img id=rOut class=out style=display:none>
  </div>
</div>

<div class="pane" id=pane-furnish>
  <div class=card>
    <div class=mut>План + текстовый бриф → меблированный фотореалистичный рендер (общий бэкенд SD+ControlNet). Опиши обстановку — модель расставит мебель по структуре плана.</div>
    <input id=fFile type=file accept="image/*">
    <textarea id=fBrief placeholder="'уютная спальня, кровать у дальней стены, две тумбы, рабочий стол у окна'"></textarea>
    <button id=fGo>Меблировать →</button>
    <div id=fStatus class=mut></div>
    <img id=fOut class=out style=display:none>
  </div>
</div>

<script type="importmap">
{"imports":{
  "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const $=s=>document.querySelector(s);
// tabs
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.pane').forEach(x=>x.classList.remove('on'));
  t.classList.add('on'); $('#pane-'+t.dataset.p).classList.add('on');
  if(t.dataset.p==='plan') onResize();
});
// status
async function refreshStatus(){
  try{
    const s=await (await fetch('api/status')).json();
    $('#disk').textContent='· диск '+s.free_gb+' ГБ свободно';
    const set=(id,ok)=>{const e=$('#st-'+id);e.textContent=ok?'готов':'нет';e.className='pill '+(ok?'ok':'no');};
    // render/furnish are on-demand (the model loads per request, then swaps back)
    const setOD=(id,ok)=>{const e=$('#st-'+id);e.textContent=ok?'активен':'по запросу';e.className='pill '+(ok?'ok':'od');};
    set('floorplan',s.backends.floorplan); setOD('render',s.backends.render); setOD('respace',s.backends.render);
  }catch(e){}
}
refreshStatus(); setInterval(refreshStatus,8000);

// ---- Three.js floor-plan -> 3D ----
let renderer,scene,camera,controls;
function init3D(){
  const el=$('#view'); if(renderer){return;}
  renderer=new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(el.clientWidth,el.clientHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.05;
  el.appendChild(renderer.domElement);
  scene=new THREE.Scene(); scene.background=new THREE.Color(0x141922);
  camera=new THREE.PerspectiveCamera(50,el.clientWidth/el.clientHeight,1,20000);
  controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true; controls.dampingFactor=0.08; controls.maxPolarAngle=Math.PI*0.49;
  scene.add(new THREE.HemisphereLight(0xeaf0ff,0x2a3340,0.85));
  const sun=new THREE.DirectionalLight(0xfff4e6,1.55); sun.position.set(900,1700,700);
  sun.castShadow=true; sun.shadow.mapSize.set(2048,2048);
  sun.shadow.camera.near=100; sun.shadow.camera.far=8000;
  const sc=3000; Object.assign(sun.shadow.camera,{left:-sc,right:sc,top:sc,bottom:-sc});
  sun.shadow.bias=-0.0004; scene.add(sun);
  const fill=new THREE.DirectionalLight(0x88aaff,0.35); fill.position.set(-600,500,-700); scene.add(fill);
  animate();
}
function animate(){requestAnimationFrame(animate); if(controls)controls.update(); if(renderer)renderer.render(scene,camera);}
function onResize(){const el=$('#view'); if(renderer&&el.clientWidth){renderer.setSize(el.clientWidth,el.clientHeight);camera.aspect=el.clientWidth/el.clientHeight;camera.updateProjectionMatrix();}}
window.addEventListener('resize',onResize);

function makeLabel(text,x,z,color,size,y){
  const F=44, pad=18, c=document.createElement('canvas'),ctx=c.getContext('2d');
  ctx.font='600 '+F+'px system-ui'; const tw=Math.ceil(ctx.measureText(text).width);
  c.width=tw+pad*2; c.height=F+pad;
  ctx.font='600 '+F+'px system-ui';
  // rounded pill background for a clean, readable chip
  const r=14,w=c.width,h=c.height; ctx.fillStyle='rgba(13,17,23,0.82)';
  ctx.beginPath(); ctx.moveTo(r,0); ctx.arcTo(w,0,w,h,r); ctx.arcTo(w,h,0,h,r); ctx.arcTo(0,h,0,0,r); ctx.arcTo(0,0,w,0,r); ctx.fill();
  ctx.strokeStyle='rgba(255,255,255,0.12)'; ctx.lineWidth=2; ctx.stroke();
  ctx.fillStyle=color; ctx.textBaseline='middle'; ctx.fillText(text,pad,h/2+2);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false,depthWrite:false,sizeAttenuation:true}));
  const s=size||48; sp.scale.set(s*c.width/c.height,s,1); sp.position.set(x,(y!=null?y:175),z); sp.renderOrder=999; scene.add(sp);
}
function build(data){
  init3D();
  // clear old meshes + labels (keep lights)
  scene.children.filter(o=>o.isMesh||o.isSprite).forEach(o=>scene.remove(o));
  const W=data.Width||1000, H=data.Height||1000, cx=W/2, cz=H/2;
  const M=Math.max(W,H), WALL_H=Math.round(M*0.11), TH=Math.max(M*0.012,6);
  // base slab + soft grid for scale
  const slab=new THREE.Mesh(new THREE.BoxGeometry(W*1.06,12,H*1.06),
    new THREE.MeshStandardMaterial({color:0x3a414e,roughness:0.95}));
  slab.position.y=-6; slab.receiveShadow=true; scene.add(slab);
  const grid=new THREE.GridHelper(M*1.06, 28, 0x4a525f, 0x2c323c); grid.position.y=0.6; scene.add(grid);
  // construction materials
  const matWall=new THREE.MeshStandardMaterial({color:0xe6e9ef,roughness:0.85,metalness:0.0});
  const matWin =new THREE.MeshPhysicalMaterial({color:0x8fc6ff,roughness:0.1,metalness:0,transmission:0.6,transparent:true,opacity:0.55});
  const matDoor=new THREE.MeshStandardMaterial({color:0xc98a4b,roughness:0.6});
  (data.points||[]).forEach((p,i)=>{
    const c=(data.classes||[])[i];
    const cls=(c&&c.name)?c.name:(typeof c==='string'?c:'wall');
    let w=Math.abs(p.x2-p.x1), d=Math.abs(p.y2-p.y1);
    // keep thin walls visibly thick along their short side
    if(cls==='wall'){ if(w<d) w=Math.max(w,TH); else d=Math.max(d,TH); }
    const h = cls==='wall'?WALL_H : cls==='window'?WALL_H*0.5 : WALL_H*0.62;
    const y = cls==='window'?WALL_H*0.5 : h/2;
    const mat = cls==='wall'?matWall : cls==='window'?matWin : matDoor;
    const m=new THREE.Mesh(new THREE.BoxGeometry(Math.max(w,3),h,Math.max(d,3)),mat);
    m.position.set((p.x1+p.x2)/2-cx, y, (p.y1+p.y2)/2-cz);
    m.castShadow=true; m.receiveShadow=true; scene.add(m);
  });
  // OCR labels — compact chips above each room; area on a second smaller chip
  const LY=WALL_H+M*0.05;
  (data.rooms||[]).forEach(r=>{
    const lx=r.x+r.w/2-cx, lz=r.y+r.h/2-cz;
    makeLabel(r.name.toUpperCase(), lx, lz, '#eef2f7', M*0.05, LY);
    if(r.area) makeLabel(r.area, lx, lz, '#7fd6a0', M*0.038, LY-M*0.05);
  });
  (data.stairs||[]).forEach(s=>{
    const lx=s.x+s.w/2-cx, lz=s.y+s.h/2-cz;
    const st=new THREE.Mesh(new THREE.BoxGeometry(M*0.05,WALL_H*0.5,M*0.05),
      new THREE.MeshStandardMaterial({color:0xe05a4a,roughness:0.5})); st.position.set(lx,WALL_H*0.25,lz);
    st.castShadow=true; scene.add(st);
    makeLabel('лестница', lx, lz, '#ff8a7a', M*0.042, WALL_H*0.7);
  });
  // pleasant isometric framing
  camera.position.set(M*0.62,M*0.78,M*0.62); controls.target.set(0,WALL_H*0.3,0);
  controls.minDistance=M*0.3; controls.maxDistance=M*3; controls.update();
}

$('#planGo').onclick=async()=>{
  const f=$('#planFile').files[0]; if(!f){alert('Выбери план');return;}
  $('#planGo').disabled=true; $('#planStatus').textContent='Детекция стен/дверей/окон…';
  const fd=new FormData(); fd.append('image',f);
  try{
    const r=await fetch('api/floorplan',{method:'POST',body:fd});
    if(!r.ok){$('#planStatus').textContent='Ошибка: '+r.status+' '+await r.text(); $('#planGo').disabled=false; return;}
    const data=await r.json();
    build(data);
    const rooms=data.rooms||[], stairs=data.stairs||[];
    $('#planStatus').textContent=`Готово: ${(data.points||[]).length} конструкций · комнат ${rooms.length} · лестниц ${stairs.length} (${data.Width}×${data.Height}).`;
    $('#recog').innerHTML = rooms.length
      ? '<b>Распознанные помещения:</b> '+rooms.map(r=>r.name+(r.area?' ('+r.area+')':'')).join(' · ')
      : 'Текст на плане не распознан (OCR ничего не нашёл).';
  }catch(e){$('#planStatus').textContent='Сбой: '+e;}
  $('#planGo').disabled=false;
};

async function postFile(url,inputId,statusId,extra){
  const f=$(inputId).files[0]; if(!f){alert('Выбери план');return null;}
  $(statusId).textContent='Обработка…';
  const fd=new FormData(); fd.append('image',f); for(const k in (extra||{}))fd.append(k,extra[k]);
  const r=await fetch(url,{method:'POST',body:fd});
  if(!r.ok){$(statusId).textContent='Ошибка: '+r.status+' '+await r.text(); return null;}
  return r;
}
$('#rGo').onclick=async()=>{
  $('#rStatus').textContent='Рендер (первый запуск качает модели, это минуты; STT/TTS на паузе)…';
  const r=await postFile('api/render','#rFile','#rStatus',{prompt:$('#rPrompt').value,preset:$('#rPreset').value});
  if(r){const b=await r.blob(); $('#rOut').src=URL.createObjectURL(b); $('#rOut').style.display='block'; $('#rStatus').textContent='Готово';}};
$('#fGo').onclick=async()=>{
  $('#fStatus').textContent='Меблировка (первый запуск качает модели, это минуты)…';
  const r=await postFile('api/furnish','#fFile','#fStatus',{brief:$('#fBrief').value});
  if(r){const b=await r.blob(); $('#fOut').src=URL.createObjectURL(b); $('#fOut').style.display='block'; $('#fStatus').textContent='Готово';}
};
</script></body></html>"""
