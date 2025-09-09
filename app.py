# app.py
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dotenv import load_dotenv
from scoring import compute_scores_and_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

load_dotenv()

app = FastAPI(title="AI Trustworthiness · Cleanlab")

# 挂载静态目录
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 模板
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"])
)

# 文件修改时间 → 版本号（用于破缓存）
def _file_ver(filename: str) -> str:
    try:
        return str(int(os.path.getmtime(os.path.join(STATIC_DIR, filename))))
    except Exception:
        return "0"

def logo_ver() -> str:
    return _file_ver("uon-logo.png")

def bg_ver() -> str:
    # 背景图文件名固定为 campus-bg.jpg（或你用同名 png/jpeg 也行，自己改下这里）
    for name in ("campus-bg.jpg", "campus-bg.jpeg", "campus-bg.png"):
        p = os.path.join(STATIC_DIR, name)
        if os.path.exists(p):
            return _file_ver(name)
    return "0"

# （可选）不给 static 强缓存
@app.middleware("http")
async def no_cache_static(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    template = jinja_env.get_template("index.html")
    return HTMLResponse(template.render(logo_ver=logo_ver(), bg_ver=bg_ver()))

@app.post("/evaluate", response_class=HTMLResponse)
async def evaluate(request: Request, question: str = Form(...), answer: str = Form(...)):
    result = await compute_scores_and_summary(question, answer)
    template = jinja_env.get_template("result.html")
    return HTMLResponse(template.render(**result, question=question, answer=answer,
                                       logo_ver=logo_ver(), bg_ver=bg_ver()))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)