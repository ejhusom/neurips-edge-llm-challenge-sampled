"""Inline styles, data and app code into single-file pages.

Outputs
  dist/index.html     standalone website (full HTML document)
  dist/artifact.html  body-only variant for publishing as a claude.ai artifact
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = lambda *p: open(os.path.join(ROOT, "src", *p), encoding="utf-8").read()
tpl, css, app = src("index.html"), src("styles.css"), src("app.js")
data = json.load(open(os.path.join(ROOT, "build", "data.json"), encoding="utf-8"))
blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
page = tpl.replace("/*__STYLE__*/", css).replace("/*__DATA__*/", blob).replace("/*__APP__*/", app)
os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
with open(os.path.join(ROOT, "dist", "artifact.html"), "w", encoding="utf-8") as f:
    f.write(page)
head, body = page.split('<header class="topbar">', 1)
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
       + head + '</head>\n<body>\n<header class="topbar">' + body + '\n</body>\n</html>\n')
with open(os.path.join(ROOT, "dist", "index.html"), "w", encoding="utf-8") as f:
    f.write(doc)
for n in ("index.html", "artifact.html"):
    print(n, f"{os.path.getsize(os.path.join(ROOT, 'dist', n)) / 1e6:.2f} MB")
