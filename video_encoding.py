"""Finalize OpenCV working videos as browser-compatible H.264 MP4."""
import os
import subprocess
from pathlib import Path


def browser_mp4(source, destination):
    import imageio_ffmpeg
    source,destination = Path(source),Path(destination)
    temporary = destination.with_name(destination.stem+'_encoding.mp4')
    try:
        result = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
            '-i',str(source),'-an','-c:v','libx264','-preset','fast','-crf','20',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],
            capture_output=True,text=True,timeout=180,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if result.returncode or not temporary.exists() or temporary.stat().st_size==0:
            raise RuntimeError('브라우저용 영상 변환 실패: '+result.stderr[-500:])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
