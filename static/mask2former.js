(()=>{'use strict';
const $=id=>document.getElementById(id);
let rows=[],selected=null,uploaded=null,result=null,busy=false;
async function json(response){const r=await response.json();if(!response.ok)throw Error(r.error||'요청 실패');return r}
function list(){
  $('files').replaceChildren();
  for(const row of rows.filter(r=>$('split').value==='all'||r.split===$('split').value)){
    const b=document.createElement('button');b.textContent=row.original.split(/[\\/]/).pop();
    b.disabled=busy;b.className=selected===row.id?'active':'';
    b.onclick=()=>{selected=row.id;uploaded=null;run()};$('files').append(b);
  }
}
function lock(value){busy=value;for(const id of ['apply','upload','mode','split'])$(id).disabled=value;list()}
function render(){
  if(!result)return;
  $('results').hidden=false;$('videoResult').hidden=true;
  $('original').src=result.images.original;$('enhanced').src=result.images.enhanced;$('truth').disabled=!result.record;
  for(const view of ['original','enhanced']){
    const r=result.result.views[view];$(view+'Result').hidden=!r;if(!r)continue;
    const hazard=$('display').value==='hazard'&&result.images[view+'_hazard_overlay'];
    const prefix=view+(hazard?'_hazard':'');
    $(view+'Overlay').src=($('truth').checked&&result.images[prefix+'_annotation'])||result.images[prefix+'_overlay'];
    $(view+'Save').href=result.images[prefix+'_transparent'];
    const name=view==='original'?'원본':'밝기 보정';
    $(view+'Caption').textContent=hazard?name+' · '+(r.hazard.status==='candidate'?'위험 단차 후보':'위험 단차 판단 보류'):name+' · '+(r.status==='stairs_region'?'계단 영역 검출':r.status==='insufficient_signal'?'영상 정보 부족 · 판단 보류':'계단 미검출')+' · 화면 면적 '+(r.stair_area_fraction*100).toFixed(1)+'%';
  }
  $('report').textContent=JSON.stringify(result.result,null,2);
}
async function video(file){
  $('results').hidden=true;$('videoResult').hidden=false;$('player').hidden=true;$('videoSave').hidden=true;$('progress').value=0;
  $('status').textContent='Mask2Former 영상 처리 중… 최대 1fps로 분석합니다.';
  const body=new FormData();body.append('video',file);body.append('engine','mask2former');
  body.append('input_view',$('mode').value==='original'?'original':'enhanced');
  const job=await json(await fetch('/api/video',{method:'POST',body}));
  while(true){
    await new Promise(resolve=>setTimeout(resolve,1000));
    const r=await json(await fetch('/api/video/'+job.id));
    $('videoStatus').textContent=`${r.seconds||0}초 · ${r.frames||0}프레임 처리`;$('progress').value=r.progress||0;
    if(r.status==='error')throw Error(r.error);
    if(r.status==='done'){
      $('player').src=r.download;$('player').hidden=false;$('videoSave').href=r.download;$('videoSave').hidden=false;
      $('status').textContent='Mask2Former 영상 처리 완료 · 왼쪽 원본 / 오른쪽 계단 영역.';return;
    }
  }
}
async function run(){
  if(busy)return;
  if(!selected&&!uploaded){$('status').textContent='사진 또는 영상을 선택하세요.';return}
  lock(true);$('status').textContent='Mask2Former 분석 중… 첫 실행은 모델 로딩으로 시간이 걸립니다.';
  try{
    if(uploaded&&(uploaded.type.startsWith('video/')||/\.(mp4|avi|mov|webm|mkv)$/i.test(uploaded.name))){await video(uploaded);return}
    let options;
    if(uploaded){const body=new FormData();body.append('image',uploaded);body.append('mode',$('mode').value);options={method:'POST',body}}
    else options={method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:selected,mode:$('mode').value})};
    result=await json(await fetch('/api/mask2former/analyze',options));
    $('filename').textContent=uploaded?uploaded.name:result.record.original.split(/[\\/]/).pop();render();
    $('status').textContent='완료 · '+result.result.device+' · '+(result.result.processing_ms/1000).toFixed(2)+'초. 정답 라벨은 모델 입력에 사용하지 않았습니다.';
  }catch(e){$('status').textContent=e.message}finally{lock(false)}
}
$('apply').onclick=run;
$('upload').onclick=()=>{
  const f=$('file').files[0];if(!f){$('status').textContent='사진 또는 영상을 선택하세요.';return}
  const isVideo=f.type.startsWith('video/')||/\.(mp4|avi|mov|webm|mkv)$/i.test(f.name);
  if(f.size>(isVideo?120:20)*1024*1024){$('status').textContent='파일 크기 제한을 초과했습니다.';return}
  uploaded=f;selected=null;run();
};
$('split').onchange=list;$('truth').onchange=render;$('display').onchange=render;
document.querySelectorAll('.pictures img').forEach(i=>i.onclick=()=>i.classList.toggle('zoom'));
(async()=>{try{
  const [catalog,state]=await Promise.all([fetch('/api/catalog').then(json),fetch('/api/mask2former/status').then(json)]);
  rows=catalog.records;list();$('status').textContent=state.ready?'준비 완료. 왼쪽 사진을 선택하거나 새 사진·영상을 업로드하세요.':'모델 파일 다운로드가 아직 완료되지 않았습니다.';
  if(state.hazard_evaluation){const tests=state.hazard_evaluation.summary.filter(r=>r.split==='test');$('hazardMetrics').textContent='기존 시험 자료의 위험 단차 위치 일치(IoU ≥ 0.5): '+tests.map(r=>(r.input==='original'?'원본':'보정')+' '+r.true_positive+'/'+r.images+'장').join(' / ')+'. 이전 개발에서 이미 확인한 자료이며, 새로운 장소에 대한 독립 검증은 아닙니다.'}
}catch(e){$('status').textContent=e.message}})();
})();
