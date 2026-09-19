const $=s=>document.querySelector(s), tiers={T1:[10,10,80],T2:[80,20,80],T3:[20,80,80],T4:[10,490,1000]};
let result=null,frame=0,angle=.4,tilt=.2,zoom=1,playing=false,lastTick=0;
const canvas=$('#scene'),ctx=canvas.getContext('2d');
function protocol(){const [k,h,dt]=tiers[$('#tier').value];$('#protocol').textContent=`${k} 帧观测 → ${h} 帧预测 · ${dt} ps / 帧`;}
$('#tier').onchange=protocol;protocol();
async function api(path,options){const r=await fetch(path,options);const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data;}
const labels={uploading:'上传中',queued:'已排队，等待计算资源',validating:'检查输入帧数、原子顺序和时间戳',running:'模型推理中',completed:'推理完成',failed:'推理失败'};
$('#form').onsubmit=async event=>{event.preventDefault();$('#submit').disabled=true;$('#status').textContent='创建推理任务…';try{
 const job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tier:$('#tier').value,ligand:$('#ligand').value})});
 for(const [selector,name]of [['#pdb','topology.pdb'],['#xtc','observations.xtc']]){const file=$(selector).files[0];if(file.size>128*1024*1024)throw Error('每个文件最大 128 MB');$('#status').textContent='上传 '+file.name;await api(`/api/jobs/${job.id}/${name}`,{method:'PUT',body:file});}
 await api(`/api/jobs/${job.id}/run`,{method:'POST'});history.replaceState(null,'','?job='+job.id);poll(job.id);
}catch(e){$('#status').textContent=e.message;$('#submit').disabled=false;}};
async function poll(id){try{const s=await api(`/api/jobs/${id}`);$('#status').textContent=labels[s.status]+(s.error?'：'+s.error:'');
 if(s.status==='completed'){result=await api(`/api/jobs/${id}/result`);frame=0;zoom=1;$('#frame').max=result.frames-1;$('#frame').value=0;$('#frame').disabled=false;$('#play').disabled=false;$('#empty').hidden=true;$('#download').hidden=false;$('#download').href=`/api/jobs/${id}/download`;$('#download').download=s.output_filename;$('#seconds').textContent=result.inference_seconds.toFixed(1);chart('#drift',result.centre_displacement_angstrom);chart('#radius',result.radius_of_gyration_angstrom);$('#submit').disabled=false;render();}
 else if(s.status==='failed'){$('#submit').disabled=false;}
 else setTimeout(()=>poll(id),3000);
}catch(e){$('#status').textContent='连接中断：'+e.message;$('#submit').disabled=false;}}
const existing=new URLSearchParams(location.search).get('job');if(existing)poll(existing);
function chart(id,values){const max=Math.max(...values),min=Math.min(...values),range=Math.max(max-min,.001);const points=values.map((v,i)=>`${30+i*510/Math.max(values.length-1,1)},${80-(v-min)/range*62}`).join(' ');$(id).innerHTML=`<path d="M30 80H545" stroke="#354648"/><polyline points="${points}" fill="none" stroke="#bded83" stroke-width="2"/><text x="0" y="20" fill="#92a5a5" font-size="10">${max.toFixed(2)}</text><text x="0" y="83" fill="#92a5a5" font-size="10">${min.toFixed(2)}</text><text x="30" y="102" fill="#92a5a5" font-size="10">首帧</text><text x="516" y="102" fill="#92a5a5" font-size="10">末帧</text>`;}
function render(){const ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;ctx.scale(ratio,ratio);if(!result)return;
 const x=result.coordinates[frame],all=[...result.protein_ca,...result.observed_last];let center=[0,0,0];for(const p of all)for(let a=0;a<3;a++)center[a]+=p[a]/all.length;const span=Math.max(...all.map(p=>Math.hypot(...p.map((v,a)=>v-center[a]))),10);const scale=Math.min(w,h)*.40/span*zoom;
 function project(p){const[a,b,c]=p.map((v,i)=>v-center[i]),u=a*Math.cos(angle)+c*Math.sin(angle),z=-a*Math.sin(angle)+c*Math.cos(angle);return[w/2+u*scale,h/2+(b*Math.cos(tilt)-z*Math.sin(tilt))*scale,b*Math.sin(tilt)+z*Math.cos(tilt)];}
 function line(a,b,color,width){a=project(a);b=project(b);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
 for(let i=1;i<result.protein_ca.length;i++){const a=result.protein_ca[i-1],b=result.protein_ca[i];if(Math.hypot(...a.map((v,j)=>v-b[j]))<5)line(a,b,'#466466',1.5);}
 for(const[a,b]of result.bonds)line(result.observed_last[a],result.observed_last[b],'#69755255',1);
 for(const[a,b]of result.bonds)line(x[a],x[b],'#b2ce9c',2);
 x.map((p,i)=>({p:project(p),e:result.elements[i]})).sort((a,b)=>a.p[2]-b.p[2]).forEach(({p,e})=>{ctx.beginPath();ctx.arc(p[0],p[1],Math.max(1.6,Math.min(6,scale*(e==='H'?.22:.45))),0,2*Math.PI);ctx.fillStyle={C:'#bded83',O:'#ef8d79',N:'#91bfff',S:'#ebd975',H:'#d2dddd'}[e]||'#c8a5dc';ctx.fill();});
 $('#counter').textContent=`${frame+1} / ${result.frames}`;$('#frameLabel').textContent=`${result.tier} · ${(result.first_time_ps+frame*result.dt_ps)/1000} ns`;}
$('#frame').oninput=e=>{frame=+e.target.value;render();};$('#play').onclick=()=>{playing=!playing;$('#play').textContent=playing?'暂停':'播放';};
let drag=null;canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(drag){angle+=(e.clientX-drag[0])*.008;tilt+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];render();}};canvas.onpointerup=()=>drag=null;canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));render();};new ResizeObserver(render).observe(canvas);
function animate(t){if(playing&&result&&t-lastTick>120){lastTick=t;frame=(frame+1)%result.frames;$('#frame').value=frame;render();}requestAnimationFrame(animate);}requestAnimationFrame(animate);
