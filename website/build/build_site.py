"""Assemble the single-file website from src/, build/data.json and vendor/.

Outputs
  docs/index.html            fully offline page for GitHub Pages: D3 and the fonts are inlined
  docs/.nojekyll             serve the file as-is, without Jekyll processing
  website/build/artifact.html  variant that loads D3 and fonts from CDNs (claude.ai preview)

Run from anywhere:  python3 website/build/build_site.py   (standard library only)
"""
import base64, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
REPO = os.path.dirname(WEB)
DOCS = os.path.join(REPO, "docs")


def read(*p, mode="r"):
    with open(os.path.join(WEB, *p), mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
        return f.read()


tpl, css, app = read("src", "index.html"), read("src", "styles.css"), read("src", "app.js")
data = json.loads(read("build", "data.json"))
blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
page = tpl.replace("/*__STYLE__*/", css).replace("/*__DATA__*/", blob).replace("/*__APP__*/", app)

# ---- CDN variant (claude.ai artifact) ----
cdn_fonts = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
    '&family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400&family=Schibsted+Grotesk:wght@400..800&display=swap">'
)
cdn_d3 = '<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>'
artifact = page.replace("<!--__FONTS__-->", cdn_fonts).replace("<!--__D3__-->", cdn_d3)
with open(os.path.join(HERE, "artifact.html"), "w", encoding="utf-8") as f:
    f.write(artifact)

# ---- offline variant (GitHub Pages) ----
faces = []
for face in json.loads(read("vendor", "fonts", "fonts.json")):
    b64 = base64.b64encode(read("vendor", "fonts", face["file"], mode="rb")).decode("ascii")
    faces.append(
        "@font-face{"
        f"font-family:'{face['family']}';font-style:{face['style']};font-weight:{face['weight']};font-display:swap;"
        f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
    )
fonts_inline = (
    "<style>/* IBM Plex Mono, Newsreader, Schibsted Grotesk: SIL Open Font License 1.1, see website/vendor/fonts */\n"
    + "\n".join(faces) + "\n</style>"
)
d3_src = read("vendor", "d3.v7.9.0.min.js")
assert "</script" not in d3_src.lower()
d3_inline = "<script>\n" + d3_src + "\n</script>"
offline = page.replace("<!--__FONTS__-->", fonts_inline).replace("<!--__D3__-->", d3_inline)
assert "fonts.googleapis.com" not in offline and "cdnjs.cloudflare.com" not in offline

head, body = offline.split('<header class="topbar">', 1)
doc = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    + head + "</head>\n<body>\n<header class=\"topbar\">" + body + "\n</body>\n</html>\n"
)
os.makedirs(DOCS, exist_ok=True)
with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
    f.write(doc)
open(os.path.join(DOCS, ".nojekyll"), "w").close()

for path in (os.path.join(DOCS, "index.html"), os.path.join(HERE, "artifact.html")):
    print(f"{os.path.relpath(path, REPO):32s} {os.path.getsize(path) / 1e6:.2f} MB")
