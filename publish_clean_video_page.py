"""Concise saved-result page without FPS clutter or live-inference claims."""
from mask2former_detector import ROOT


def main():
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>영상 분석 결과</title><style>body{font:18px/1.8 sans-serif;max-width:1400px;margin:30px auto;padding:20px;background:#14212e;color:white}video{width:100%;max-height:80vh;background:black}a{color:#8ee6ff}.note{font-size:14px;color:#aabccd}</style><a href="/">← 대시보드</a><h1>영상 분석 결과</h1><p>왼쪽 원본 · 오른쪽 밝기 보정 및 단차 후보</p><p class="note">dark.mp4 · 저장된 분석 결과 · 원본 9초부터 끝까지</p><video controls preload="metadata" src="source_comparison_10fps.mp4"></video><p><a href="source_comparison_10fps.mp4" download="dark_analysis_result.mp4">결과 영상 다운로드</a></p><details><summary>분석 결과만 보기</summary><video controls preload="none" src="overlay_10fps.mp4"></video><p><a href="overlay_10fps.mp4" download="dark_overlay.mp4">표시 영상 다운로드</a></p></details></html>'''
    (ROOT/'outputs/dark_demo_orange/10fps.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
