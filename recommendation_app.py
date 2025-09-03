import os, json, unicodedata
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate
from dash import dash_table

ART_ROOT  = Path(os.environ.get("ART_ROOT", "artifacts/psyllabus")).resolve()
META_JSON = Path("../Piano syllabus/final_index/new_clean_data.json").resolve()

def pick_first(folder: Path, names):
    for n in names:
        p = folder / n
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {names} found under {folder}")

def npy(path: Path):
    return np.load(path, allow_pickle=True)

def l2norm_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def normalize_key(s: str) -> str:
    """Lowercase, strip accents, unify separators, keep letters/digits/spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    for t in ["_", ".", "-", "/", ":", ";"]:
        s = s.replace(t, " ")
    s = " ".join(s.split())
    return "".join(ch for ch in s if ch.isalnum() or ch == " ")

def to_embed_url(u: str) -> str:
    """Convert watch/short/shortened YouTube URLs to embeddable form."""
    u = (u or "").strip()
    if not u:
        return ""
    if "watch?v=" in u:
        vid = u.split("watch?v=")[1].split("&")[0]
        return f"https://www.youtube.com/embed/{vid}"
    if "youtu.be/" in u:
        vid = u.split("youtu.be/")[1].split("?")[0]
        return f"https://www.youtube.com/embed/{vid}"
    if "/shorts/" in u:
        vid = u.split("/shorts/")[1].split("?")[0].split("&")[0]
        return f"https://www.youtube.com/embed/{vid}"
    return u  # already an embed or playlist url

def yt_search_embed(title: str, comp: str) -> str:
    q = quote_plus(f"{title} {comp} piano")
    return f"https://www.youtube.com/embed?listType=search&list={q}"

# Load metadata
if not META_JSON.exists():
    raise SystemExit(f"Metadata JSON not found: {META_JSON}")
META_RAW = json.loads(Path(META_JSON).read_text(encoding="utf-8"))
META_BY_NORMKEY = {normalize_key(k): v for k, v in META_RAW.items()}

def meta_for_id(piece_id: str):
    if piece_id in META_RAW:
        return META_RAW[piece_id]
    nid = normalize_key(piece_id)
    if nid in META_BY_NORMKEY:
        return META_BY_NORMKEY[nid]
    for nk, v in META_BY_NORMKEY.items():
        if nk and nk in nid:
            return v
    return {}


# Artifact folders
if not ART_ROOT.exists():
    raise SystemExit(f"Artifacts root not found: {ART_ROOT}")

MODEL_DIRS = []
for d in ART_ROOT.iterdir():
    if not d.is_dir():
        continue
    try:
        ids_p    = pick_first(d, ["ids.npy"])
        vecs_p   = pick_first(d, ["vecs.npy", "vectors.npy", "X.npy"])
        coords_p = pick_first(d, ["coords.npy", "coords_umap.npy", "umap_2d.npy", "umap.npy"])
        MODEL_DIRS.append(dict(
            name=d.name, folder=d.resolve(),
            ids_p=ids_p, vecs_p=vecs_p, coords_p=coords_p
        ))
    except FileNotFoundError:
        continue

if not MODEL_DIRS:
    raise SystemExit(f"No valid artifact folders under {ART_ROOT}")

REGISTRY = {}  
for m in MODEL_DIRS:
    ids = npy(m["ids_p"]).astype(str)
    X   = npy(m["vecs_p"]).astype(np.float32)
    Y2  = npy(m["coords_p"]).astype(np.float32)

    if np.abs(np.linalg.norm(X, axis=1) - 1).max() > 1e-3:
        X = l2norm_rows(X)

    titles, composers, eras, diffs, yts = [], [], [], [], []
    for pid in ids:
        meta = meta_for_id(pid)
        title = meta.get("PS_title") or meta.get("title") or pid
        comp  = meta.get("composer", "")
        era   = meta.get("period") or meta.get("era") or "-"
        diff  = meta.get("ps_rating") or meta.get("ps") or meta.get("difficulty")
        try:
            diff = int(diff) if diff not in (None, "") else None
        except Exception:
            diff = None
        yt    = to_embed_url(meta.get("youtube_link") or meta.get("youtube") or meta.get("YT_link") or meta.get("link") or "")
        titles.append(title); composers.append(comp); eras.append(era); diffs.append(diff); yts.append(yt)

    df = pd.DataFrame(dict(
        id=ids,
        x=Y2[:,0].astype(float),
        y=Y2[:,1].astype(float),
        idx=np.arange(len(ids), dtype=int),
        title=titles, composer=composers, era=eras, difficulty=diffs, youtube=yts
    ))
    REGISTRY[m["name"]] = dict(df=df, vecs=X, folder=str(m["folder"]))

MODEL_NAMES   = sorted(REGISTRY.keys())
DEFAULT_MODEL = ("audio_midi_pianoroll_ps_5_v4_post"
                 if "audio_midi_pianoroll_ps_5_v4_post" in REGISTRY else MODEL_NAMES[0])


# Cosine similarity
def cosine_topk(vecs: np.ndarray, seed_idx: int, k: int = 25):
    v = vecs[seed_idx:seed_idx+1]
    sims = (vecs @ v.T).ravel()            
    k = min(k, len(sims))
    take = np.argpartition(-sims, range(k))[:k]
    order = take[np.argsort(-sims[take])]
    return order, sims[order]

## App
app = Dash(__name__)
app.title = "Embeddings exploration and recommendation of classical piano recordings"

def model_options():
    return [{"label": m, "value": m} for m in MODEL_NAMES]

def seed_options(df: pd.DataFrame):
    dd = df.sort_values("title")
    return [{"label": f"{r['title']} — {r['composer']}", "value": int(r["idx"])} for _, r in dd.iterrows()]

def scatter_figure(df: pd.DataFrame, color_by: str, palette_mode: str):
    """palette_mode: 'normal' | 'blue' (single-hue gradient)."""
    hover = {"composer": True, "era": True, "difficulty": True}
    custom = ["idx", "youtube", "title", "composer"]

    if palette_mode == "blue":
        series = df[color_by] if color_by in df.columns else None
        if series is None:
            fig = px.scatter(df, x="x", y="y", hover_name="title",
                             hover_data=hover, custom_data=custom, height=650)
        else:
            if pd.api.types.is_numeric_dtype(series):
                col = series
            else:
                col = pd.Categorical(series).codes
                if (col < 0).any():
                    col = np.where(col < 0, col.max() + 1, col)
            fig = px.scatter(df, x="x", y="y", color=col, color_continuous_scale="Blues",
                             hover_name="title", hover_data=hover, custom_data=custom, height=650)
            fig.update_layout(coloraxis_colorbar_title=color_by)
    else:
        fig = px.scatter(df, x="x", y="y",
                         color=color_by if color_by in df.columns else None,
                         hover_name="title", hover_data=hover,
                         custom_data=custom, height=650)

    fig.update_layout(margin=dict(l=10, r=10, b=10, t=40))
    fig.update_traces(marker=dict(size=7, opacity=0.9))
    return fig

SPLIT_ROW_STYLE   = {"display": "flex", "gap": "12px", "alignItems": "flex-start"}
LEFT_COL_STYLE    = {"flex": "0 0 66%", "display": "block"}                   # plot (shown in explore)
RIGHT_COL_STYLE   = {"flex": "1 1 34%", "display": "block", "minWidth": 0}    # player + table (explore)
FULL_RIGHT_STYLE  = {"flex": "1 1 100%", "display": "block", "minWidth": 0}   # full-width for recommendation only

app.layout = html.Div([
    html.H2("Embeddings exploration and recommendation of classical piano recordings"),

    html.Div([
        dcc.Checklist(
            id="mode-toggle",
            options=[{"label": " Explore embeddings", "value": "explore"}],
            value=["explore"], #default
            inputStyle={"marginRight": "6px"},
            style={"marginBottom": "8px"}
        )
    ]),

    html.Div([
        html.Div([
            html.Label("Model"),
            dcc.Dropdown(id="model-dd", options=model_options(), value=DEFAULT_MODEL, clearable=False),
        ], id="model-wrap", style={"width":"22%","display":"inline-block","paddingRight":"10px"}),

        html.Div([
            html.Label("Choose piece"),
            dcc.Dropdown(id="seed-dd", options=[], value=None, placeholder="Type to search a piece…"),
        ], style={"width":"40%","display":"inline-block","paddingRight":"10px"}),

        html.Div([
            html.Label("Color by"),
            dcc.Dropdown(id="color-by",
                         options=[{"label":"difficulty","value":"difficulty"},
                                  {"label":"era","value":"era"}],
                         value="difficulty", clearable=False),
        ], id="colorby-wrap", style={"width":"18%","display":"inline-block","paddingRight":"10px"}),

        html.Div([
            html.Label("Color style"),
            dcc.Dropdown(
                id="palette-mode",
                options=[{"label":"Normal","value":"normal"},
                         {"label":"Blue gradient","value":"blue"}],
                value="normal", clearable=False
            )
        ], id="palette-wrap", style={"width":"18%","display":"inline-block","paddingRight":"10px"}),

        html.Div([
            html.Label("Top recommendations:"),
            dcc.Slider(id="k-slider", min=5, max=50, step=1, value=20,
                       marks={5:"5", 20:"20", 35:"35", 50:"50"})
        ], style={"width":"20%","display":"inline-block","verticalAlign":"top"}),
    ], style={"paddingBottom":"8px"}),

    # left (plot) + right (player/table)
    html.Div([
        html.Div([ dcc.Graph(id="scatter", figure={}, clear_on_unhover=True) ],
                 id="graph-wrap", style=LEFT_COL_STYLE),

        html.Div([
            html.Iframe(id="yt-frame", src="", style={"width":"100%","height":"360px","border":"0"}),
            html.Div(id="nowplaying", style={"marginTop":"6px","fontWeight":"600"}),
            html.Hr(),
           dash_table.DataTable(
    id="neighbors",
    columns=[
        {"name": "rank",       "id": "rank"},
        {"name": "similarity", "id": "similarity", "type": "numeric",
         "format": {"specifier": ".3f"}},
        {"name": "title",      "id": "title"},
        {"name": "composer",   "id": "composer"},
        {"name": "era",        "id": "era"},
        {"name": "difficulty", "id": "difficulty"},
    ],
    data=[],
    page_size=15,
    fill_width=True,
    style_table={
        "height": "420px",
        "overflowY": "auto",
        "backgroundColor": "rgba(255,255,255,0.96)",
        "border": "1px solid #e5e7eb",
        "borderRadius": "8px",
    },
    style_header={
        "backgroundColor": "#f5f7fb",
        "fontWeight": "600",
        "color": "#0b132b",
        "borderBottom": "1px solid #e5e7eb",
    },
    style_cell={
        "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        "fontSize": 13,
        "padding": "8px",
        "whiteSpace": "normal",
        "height": "auto",
        "color": "#111827",
        "backgroundColor": "rgba(255,255,255,0.96)",
    },
    style_data={
        "color": "#111827",
        "backgroundColor": "rgba(255,255,255,0.96)",
        "border": "0px",
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"},
         "backgroundColor": "rgba(247,249,252,0.96)"},
        {"if": {"state": "selected"},
         "backgroundColor": "#e6f0ff",
         "color": "#111827",
         "border": "1px solid #3b82f6"},
        {"if": {"state": "active"},
         "border": "1px solid #3b82f6"},
    ],
    css=[  
        {"selector": ".dash-spreadsheet td div",
         "rule": "max-height: 3.2em; overflow: hidden; text-overflow: ellipsis;"},
    ],
    style_as_list_view=True,
    row_selectable="single",
    selected_rows=[],
)

        ], id="right-wrap", style=RIGHT_COL_STYLE),
    ], id="split-row", style=SPLIT_ROW_STYLE),

    dcc.Store(id="store-model-name"),
    dcc.Store(id="store-seed-idx"),
])

# hide/show plot and exploration controls and expand right side based on the mode
@app.callback(
    [Output("graph-wrap", "style"),
     Output("model-wrap", "style"),
     Output("colorby-wrap", "style"),
     Output("palette-wrap", "style"),
     Output("right-wrap", "style")],
    Input("mode-toggle", "value")
)
def toggle_explore(val):
    explore = "explore" in (val or [])
    # left panel visible only in explore
    left_style = LEFT_COL_STYLE if explore else {"display": "none"}
    # controls visible only in explore
    show_ctrl   = {"width":"22%","display":"inline-block","paddingRight":"10px"}
    show_color  = {"width":"18%","display":"inline-block","paddingRight":"10px"}
    hide        = {"display":"none"}
    # split in explore and full width otherwise
    right_style = RIGHT_COL_STYLE if explore else FULL_RIGHT_STYLE

    return (
        left_style,
        show_ctrl if explore else hide,
        show_color if explore else hide,
        show_color if explore else hide,
        right_style
    )

# Initialize the app
@app.callback(
    [Output("seed-dd","options"),
     Output("seed-dd","value"),
     Output("scatter","figure"),
     Output("store-model-name","data")],
    [Input("model-dd","value"),
     Input("color-by","value"),
     Input("palette-mode","value")],
    prevent_initial_call=False
)
def init_model(model_name, color_by, palette_mode):
    if model_name is None:
        raise PreventUpdate
    df = REGISTRY[model_name]["df"]
    default_seed = int(df.iloc[0]["idx"]) if len(df) else None
    fig = scatter_figure(df, color_by, palette_mode)
    return seed_options(df), default_seed, fig, model_name

# Exclude the seed itself from ranking
@app.callback(
    Output("neighbors","data"),
    [Input("store-model-name","data"),
     Input("seed-dd","value"),
     Input("k-slider","value")],
    prevent_initial_call=True
)
def recompute_neighbors(model_name, seed_idx, k):
    if not model_name or seed_idx is None:
        raise PreventUpdate
    bundle = REGISTRY[model_name]
    df, vecs = bundle["df"], bundle["vecs"]
    k = int(k or 20)
    k = min(max(k,1), len(df))

    order, sims = cosine_topk(vecs, int(seed_idx), k=k+1) 
    keep = [(i, s) for (i, s) in zip(order, sims) if int(i) != int(seed_idx)]
    keep = keep[:k]

    rows = []
    for rank, (i, s) in enumerate(keep, start=1):
        r = df.iloc[int(i)]
        rows.append(dict(
            rank=rank, similarity=float(s),
            title=str(r["title"]), composer=str(r["composer"]),
            era=str(r["era"]),
            difficulty=int(r["difficulty"]) if pd.notna(r["difficulty"]) else None,
            idx=int(r["idx"]), youtube=str(r["youtube"]),
        ))
    return rows

# autoselect first neighbor
@app.callback(
    Output("neighbors", "selected_rows"),
    [Input("neighbors", "data"),
     Input("seed-dd", "value")],
    prevent_initial_call=True
)
def autoselect_first_row(table_data, seed_val):
    if table_data and len(table_data) > 0:
        return [0]
    raise PreventUpdate

@app.callback(
    [Output("yt-frame","src"),
     Output("nowplaying","children"),
     Output("store-seed-idx","data")],
    [Input("scatter","clickData"),
     Input("neighbors","active_cell"),
     Input("neighbors","selected_rows"),
     Input("neighbors","data"),
     Input("seed-dd","value"),
     Input("store-model-name","data")],
    prevent_initial_call=True
)
def play_media(sc_click, active_cell, selected_rows, table_data, seed_value, model_name):
    trig = ctx.triggered_id

    if trig == "scatter" and sc_click:
        p = sc_click["points"][0]
        yt = p["customdata"][1] or ""
        title = p["customdata"][2]; comp = p["customdata"][3]
        if not yt:
            yt = yt_search_embed(title, comp)
        label = f"▶️ {title} — {comp}"
        seed_idx = int(p["customdata"][0])
        return yt, label, seed_idx

    # from neighbors table (clicked cell or selected row)
    if trig == "neighbors":
        row_ix = None
        if active_cell and isinstance(active_cell, dict):
            row_ix = active_cell.get("row")
        if (row_ix is None) and selected_rows:
            row_ix = selected_rows[0]
        if row_ix is not None and table_data and 0 <= row_ix < len(table_data):
            r = table_data[row_ix]
            yt = r.get("youtube","") or ""
            title = r.get("title",""); comp = r.get("composer","")
            if not yt:
                yt = yt_search_embed(title, comp)
            label = f"▶️ {title} — {comp}"
            seed_idx = int(r.get("idx"))
            return yt, label, seed_idx

    if trig == "seed-dd" and model_name is not None and seed_value is not None:
        df = REGISTRY[model_name]["df"]
        rr = df.loc[df["idx"] == int(seed_value)]
        if len(rr):
            r = rr.iloc[0]
            yt = r.get("youtube","") or ""
            title = str(r.get("title","")); comp = str(r.get("composer",""))
            if not yt:
                yt = yt_search_embed(title, comp)
            label = f"▶️ {title} — {comp}"
            return yt, label, int(seed_value)
        return no_update, no_update, seed_value

    raise PreventUpdate

if __name__ == "__main__":
    print(f"[reco] using artifacts at: {ART_ROOT}")
    print(f"[reco] metadata json:      {META_JSON}")
    print("[reco] models found:", ", ".join(MODEL_NAMES))
    app.run(debug=True, host="0.0.0.0", port=8050)
