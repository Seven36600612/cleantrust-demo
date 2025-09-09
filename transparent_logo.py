# transparent_logo.py
# 将 static/uon-logo.png 的白色/近白背景（与边缘相连部分）抠除为透明
from pathlib import Path
import numpy as np

def ensure_pillow():
    try:
        import PIL  # noqa
    except Exception:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow>=10.0.0"])
    from PIL import Image  # noqa

def remove_edge_background(src: Path, dst: Path, thr: int = 24, feather: float = 0.6):
    from PIL import Image, ImageFilter
    img = Image.open(src).convert("RGBA")
    arr = np.asarray(img).copy()
    h, w = arr.shape[0], arr.shape[1]
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]

    # 估计背景色：四角的均值
    corners = np.vstack([rgb[0,0], rgb[0,-1], rgb[-1,0], rgb[-1,-1]])
    bg = corners.mean(axis=0)

    # 与背景色的距离
    diff = np.sqrt(((rgb - bg)**2).sum(axis=2))
    near_bg = diff < thr

    # 只移除“与边缘连通”的近背景区域（避免把内部白色文字也挖掉）
    from collections import deque
    vis = np.zeros((h, w), dtype=bool)
    dq = deque()

    # 把边缘上属于 near_bg 的点入队
    for x in range(w):
        if near_bg[0, x]: vis[0, x] = True; dq.append((0, x))
        if near_bg[h-1, x]: vis[h-1, x] = True; dq.append((h-1, x))
    for y in range(h):
        if near_bg[y, 0]: vis[y, 0] = True; dq.append((y, 0))
        if near_bg[y, w-1]: vis[y, w-1] = True; dq.append((y, w-1))

    # 4-连通 BFS
    for (dy, dx) in [(1,0), (-1,0), (0,1), (0,-1)]:
        pass  # 仅为阅读方便
    from collections import deque as _dq
    dq = _dq([(y, x) for y in range(h) for x in [0, w-1] if near_bg[y, x]] +
             [(y, x) for x in range(w) for y in [0, h-1] if near_bg[y, x]])
    vis = np.zeros((h, w), dtype=bool)
    for y, x in list(dq):
        vis[y, x] = True
    while dq:
        y, x = dq.popleft()
        for ny, nx in ((y+1,x),(y-1,x),(y,x+1),(y,x-1)):
            if 0 <= ny < h and 0 <= nx < w and not vis[ny, nx] and near_bg[ny, nx]:
                vis[ny, nx] = True
                dq.append((ny, nx))

    # 将连通背景处 alpha 设为 0，其他保持
    new_alpha = np.where(vis, 0, alpha)

    # 羽化边缘让过渡更自然
    from PIL import Image
    a_img = Image.fromarray(new_alpha.astype("uint8"), "L")
    if feather > 0:
        a_img = a_img.filter(ImageFilter.GaussianBlur(feather))
    arr[..., 3] = np.array(a_img)

    out = Image.fromarray(arr, "RGBA")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, "PNG", optimize=True)

if __name__ == "__main__":
    ensure_pillow()
    src = Path("/Users/mt/PythonProject3/cleantrust_app/static/uon-logo.png")
    if not src.exists():
        raise SystemExit(f"未找到：{src.resolve()}")
    bak = src.with_suffix(".bak.png")
    if not bak.exists():
        bak.write_bytes(src.read_bytes())
        print(f"已备份到 {bak}")
    remove_edge_background(src, src)  # 就地覆盖
    print(f"✅ 已写回透明版：{src.resolve()}")