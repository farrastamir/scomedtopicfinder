import streamlit as st
import pandas as pd
import zipfile
import urllib.request
import re, textwrap
from collections import Counter

# ==================================================
# ⚙️  CONFIG
# ==================================================
st.set_page_config(layout="wide")
st.title("📰 Topic Summary NoLimit Dashboard — ONM & Sosmed")

# ==================================================
# 📦  INGEST ZIP → DataFrame
# ==================================================
@st.cache_data(show_spinner=False)
def extract_csv_from_zip(zip_file):
    """Gabungkan semua *.csv di ZIP ke satu DataFrame (semua kolom string)."""
    try:
        with zipfile.ZipFile(zip_file, "r") as z:
            csvs = [f for f in z.namelist() if f.endswith(".csv")]
            if not csvs:
                st.error("❌ ZIP tidak berisi file .csv.")
                return pd.DataFrame()

            dfs = []
            for fn in csvs:
                with z.open(fn) as fh:
                    dfs.append(pd.read_csv(
                        fh, delimiter=";", quotechar='"',
                        engine="python", on_bad_lines="skip", dtype=str))
            return pd.concat(dfs, ignore_index=True)
    except zipfile.BadZipFile:
        st.error("❌ ZIP tidak valid.")
        return pd.DataFrame()

def trim_columns(df: pd.DataFrame, needed: set):
    keep = [c for c in df.columns if c.lower() in needed]
    missing = needed - {c.lower() for c in keep}
    for m in missing:
        df[m] = ""
    return df[keep + list(missing)]

# ==================================================
# 🔍  ADVANCED KEYWORDS
# ==================================================
def parse_advanced_keywords(q: str):
    q = q.strip()
    if not q:
        return [], [], []
    inc, phr, exc = [], [], []
    for tok in re.findall(r'\"[^\"]+\"|\([^)]+\)|\S+', q):
        if tok.startswith('"') and tok.endswith('"'):
            phr.append(tok[1:-1])
        elif tok.startswith('-'):
            exc.extend(tok[1:].strip('()').split())
        elif tok.startswith('(') and tok.endswith(')'):
            inc.append([w.strip() for w in tok[1:-1].split('OR') if w.strip()])
        else:
            inc.append([tok.strip()])
    return inc, phr, exc

def match_advanced(text, inc, phr, exc):
    low = text.lower()
    if any(w in low for w in exc):
        return False
    if any(p not in low for p in phr):
        return False
    return all(any(w in low for w in grp) for grp in inc)

# ==================================================
# ☁️  WORD ‑ CLOUD
# ==================================================
@st.cache_data(show_spinner=False)
def load_stopwords():
    url = ("https://raw.githubusercontent.com/stopwords-iso/"
           "stopwords-id/master/stopwords-id.txt")
    try:
        return set(pd.read_csv(url, header=None)[0].tolist())
    except Exception:
        return set()
STOPWORDS = load_stopwords()

def build_word_freq(series, n=500):
    tokens = re.findall(r"\b\w{3,}\b", " ".join(series).lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    return Counter(tokens).most_common(n)

# ==================================================
# 🎨  SENTIMENT BADGE
# ==================================================
def sentiment_badge(val):
    s = str(val).lower()
    colors = {"positive": "green", "negative": "red", "neutral": "gray"}
    if s in colors:
        return f'<span style="color:{colors[s]};font-weight:bold">{s}</span>'
    return val

# ==================================================
# 📊  ONLINE NEWS (ONM) DASHBOARD
# ==================================================
def run_onm(df: pd.DataFrame):
    df = trim_columns(df, {"title","body","url","tier","sentiment","label"})
    for c in ["title","body","url","sentiment"]:
        df[c] = df[c].astype(str).str.strip("'")
    df["label"].fillna("", inplace=True)
    df["tier"].fillna("-", inplace=True)

    st.sidebar.markdown("## 🎛️ Filter — ONM")
    stats_box = st.sidebar.empty()

    sent_sel = st.sidebar.selectbox(
        "Sentimen", ["All"] + sorted(df["sentiment"].str.lower().unique()))
    lab_sel = st.sidebar.selectbox(
        "Label", ["All"] +
        sorted({l.strip() for sub in df["label"] for l in str(sub).split(',') if l.strip()}))

    if st.sidebar.button("🔄 Clear", key="onm_clear"):
        st.session_state.update(onm_kw="", onm_hl="")
    kw = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="onm_kw")
    hl = st.sidebar.text_input("Highlight Kata", key="onm_hl")

    # ── FILTERING
    mask = pd.Series(True, index=df.index)
    if sent_sel != "All":
        mask &= df["sentiment"].str.lower() == sent_sel
    if lab_sel != "All":
        mask &= df["label"].apply(
            lambda x: lab_sel in [s.strip() for s in str(x).split(',')])
    inc, phr, exc = parse_advanced_keywords(kw)
    if kw:
        m_title = df["title"].apply(lambda x: match_advanced(x, inc, phr, exc))
        m_body  = df["body"].apply(lambda x: match_advanced(x, inc, phr, exc))
        mask &= (m_title | m_body)
    filt = df[mask]

    # ── STATISTIC
    sl = filt["sentiment"].str.lower()
    stats_box.markdown(
        f"<div style='text-align:center;font-weight:bold;font-size:20px;'>📊 Statistik</div>"
        f"<div style='font-size:28px;font-weight:bold'>Article: {len(filt):,}</div>"
        f"<div style='font-size:15px;'>"
        f"<span style='color:green;'>🟢 {(sl=='positive').sum()}</span> | "
        f"<span style='color:gray;'>⚪ {(sl=='neutral').sum()}</span> | "
        f"<span style='color:red;'>🔴 {(sl=='negative').sum()}</span></div>",
        unsafe_allow_html=True)

    # ── LINK PRIORITY
    rank = {"tier 1":0,"tier 2":1,"tier 3":2,"-":3,"":4}
    def pick_link(g):
        g = g.copy()
        g["rk"] = g["tier"].str.lower().map(rank).fillna(4)
        g.sort_values("rk", inplace=True)
        return g["url"].iloc[0] if not g["url"].empty else "-"

    grouped = (filt.groupby("title", sort=False)
               .apply(pick_link).to_frame("Link")
               .join(filt.groupby("title").agg(
                     Total=("title","size"),
                     Sentiment=("sentiment",
                                lambda x: x.mode().iloc[0] if not x.mode().empty else "-")),
                     how="left")
               .reset_index().sort_values("Total", ascending=False))

    grouped["Sentiment"] = grouped["Sentiment"].apply(sentiment_badge)
    grouped["Link"] = grouped["Link"].apply(
        lambda u: f'<a href="{u}" target="_blank">Link</a>' if u and u!="-"
        else "-")

    # highlight
    hl_set = {h.strip('"').lower()
              for h in re.findall(r'"[^"]+"|\S+', hl)}
    if hl_set:
        pat = re.compile(f"(?i)({'|'.join(map(re.escape, hl_set))})")
        grouped["title"] = grouped["title"].apply(
            lambda t: pat.sub(r"<mark>\\1</mark>", t))

    # ── DISPLAY
    col1, col2 = st.columns([0.72, 0.28])
    with col1:
        st.markdown("### 📊 Ringkasan Topik (ONM)")
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(grouped.to_html(escape=False, index=False),
                 unsafe_allow_html=True)
    with col2:
        if st.checkbox("WordCloud", key="wc_onm"):
            wc_df = pd.DataFrame(
                build_word_freq(filt["title"] + " " + filt["body"]),
                columns=["Kata", "Jumlah"])
            st.dataframe(wc_df, use_container_width=True)

# ==================================================
# 📊  SOCIAL MEDIA DASHBOARD
# ==================================================
def run_sosmed(df: pd.DataFrame):
    need = {"content","post_type","final_sentiment",
            "specific_resource","reply_to_original_id",
            "original_id","link"}
    df = trim_columns(df, need)
    df.columns = [c.lower() for c in df.columns]
    df["content"] = df["content"].astype(str)

    # Sidebar
    st.sidebar.markdown("## 🎛️ Filter — Sosmed")
    stat_box = st.sidebar.empty()

    scope = st.sidebar.radio(
        "Conversation", ("All","Tanpa Comment","Tanpa Post"), index=0)
    plats = sorted({p for p in df["specific_resource"].dropna()
                    if str(p).strip()})
    plat_sel = st.sidebar.multiselect(
        "Platform", plats, default=plats) if plats else []

    sent_sel = st.sidebar.selectbox(
        "Sentimen",
        ["All"] + sorted(df["final_sentiment"].dropna().str.lower().unique()))

    if st.sidebar.button("🔄 Clear", key="soc_clear"):
        st.session_state.update(soc_kw="", soc_hl="")
    kw = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="soc_kw")
    hl = st.sidebar.text_input("Highlight Kata", key="soc_hl")
    show_wc = st.sidebar.checkbox("WordCloud", key="soc_wc", value=True)

    # FILTER
    mask = pd.Series(True, index=df.index)
    if plat_sel: mask &= df["specific_resource"].isin(plat_sel)
    if sent_sel != "All":
        mask &= df["final_sentiment"].str.lower() == sent_sel
    inc, phr, exc = parse_advanced_keywords(kw)
    if kw:
        mask &= df["content"].apply(lambda x: match_advanced(x, inc, phr, exc))
    filt = df[mask]

    # Stats
    sl = filt["final_sentiment"].str.lower()
    stat_box.markdown(
        f"<div style='text-align:center;font-weight:bold;font-size:20px;'>📊 Statistik</div>"
        f"<div style='font-size:28px;font-weight:bold'>Talk: {len(filt):,}</div>"
        f"<div style='font-size:15px;'>"
        f"<span style='color:green;'>🟢 {(sl=='positive').sum()}</span> | "
        f"<span style='color:gray;'>⚪ {(sl=='neutral').sum()}</span> | "
        f"<span style='color:red;'>🔴 {(sl=='negative').sum()}</span></div>",
        unsafe_allow_html=True)

    # Scope logic
    is_com  = filt["reply_to_original_id"].notna()
    is_post = filt["post_type"].str.lower() == "post_made"

    if scope == "Tanpa Comment": filt_scope = filt[~is_com]
    elif scope == "Tanpa Post":  filt_scope = filt[~is_post]
    else:                        filt_scope = filt

    # Engagement measure
    comments = filt_scope[is_com]
    com_cnt  = comments.groupby("reply_to_original_id").size()
    com_sent = comments.groupby("reply_to_original_id")["final_sentiment"]\
                       .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    posts = filt_scope[is_post].copy()
    posts["value"] = posts["original_id"].map(com_cnt).fillna(0).astype(int)
    posts["sent_final"] = posts["original_id"].map(com_sent)\
                             .fillna(posts["final_sentiment"])
    posts["Type"] = "Post"

    talk = filt_scope[~is_post].copy()
    talk["value"] = 1
    talk["sent_final"] = talk["final_sentiment"]
    talk["Type"] = "Talk"

    if scope == "Tanpa Post":       base = talk
    elif scope == "Tanpa Comment":  base = pd.concat([talk, posts], ignore_index=True)
    else:                           base = pd.concat([filt_scope.assign(value=1),
                                                       posts], ignore_index=True)

    # Helpers
    def first_link(series):
        valid = series.dropna().astype(str)
        valid = valid[valid.str.strip() != ""]
        return valid.iloc[0] if not valid.empty else "-"

    def shorten(txt, n=120):
        clean = re.sub(r'<.*?>','',txt)
        return textwrap.shorten(clean, width=n, placeholder="…")

    # SUMMARY
    summary = (base.groupby("content", sort=False)
               .agg(Total=("value","sum"),
                    Sentiment=("sent_final",
                               lambda x: x.mode().iloc[0] if not x.mode().empty else "-"),
                    Platform=("specific_resource",
                              lambda x:', '.join({p.strip() for p in x.dropna() if p.strip()})),
                    LinkSample=("link", first_link),
                    Type=("Type","first"))
               .sort_values("Total", ascending=False)
               .reset_index())

    summary["Sentiment"] = summary["Sentiment"].apply(sentiment_badge)
    summary["content"] = summary["content"].apply(shorten)
    summary["LinkSample"] = summary["LinkSample"].apply(
        lambda u: f'<a href="{u}" target="_blank">Link</a>' if u and u!="-"
        else "-")

    # highlight in content
    hl_set = {h.strip('"').lower()
              for h in re.findall(r'"[^"]+"|\S+', hl)}
    if hl_set:
        pat = re.compile(f"(?i)({'|'.join(map(re.escape, hl_set))})")
        summary["content"] = summary["content"].apply(
            lambda t: pat.sub(r"<mark>\\1</mark>", t))

    # DISPLAY
    col1, col2 = st.columns([0.72,0.28])
    with col1:
        st.markdown("### 📊 Ringkasan Topik (Sosmed)")
        st.caption(f"Dataset: {len(filt):,} baris | Summary: {len(summary):,}")
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(summary.to_html(escape=False,index=False),
                 unsafe_allow_html=True)
    with col2:
        if show_wc:
            wc_source = base if scope != "Tanpa Comment" else filt_scope
            wc_df = pd.DataFrame(
                build_word_freq(wc_source["content"], 500),
                columns=["Kata","Jumlah"])
            st.markdown("### ☁️ Word Cloud (Top 500)")
            st.dataframe(wc_df, use_container_width=True)

# ==================================================
# 🚀  ENTRY POINT
# ==================================================
st.markdown("### 📁 Pilih sumber data")
inp_mode = st.radio("Input ZIP via:", ["Upload File","Link Download"], horizontal=True)

zip_data = None
if inp_mode == "Upload File":
    up = st.file_uploader("Unggah file ZIP", type="zip")
    if up: zip_data = up
else:
    url = st.text_input("Masukkan URL file ZIP")
    if st.button("Proceed") and url:
        try:
            tmp="/tmp/data.zip"; urllib.request.urlretrieve(url,tmp); zip_data = tmp
        except Exception as e:
            st.error(f"❌ Gagal mengunduh: {e}")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = None

if zip_data:
    with st.spinner("📖 Memproses data …"):
        df_all = extract_csv_from_zip(zip_data)
        if df_all.empty:
            st.error("❌ DataFrame kosong atau gagal dibaca.")
        else:
            st.session_state["last_df"] = df_all.copy()

if st.session_state["last_df"] is not None:
    df_loaded = st.session_state["last_df"]
    cols_lc = [c.lower() for c in df_loaded.columns]
    if "tier" in cols_lc:          # ONM
        run_onm(df_loaded.copy())
    elif {"content","post_type","final_sentiment"}.issubset(cols_lc):
        run_sosmed(df_loaded.copy())  # Sosmed
    else:
        st.error("❌ Struktur kolom tidak dikenali sebagai ONM maupun Sosmed.")
else:
    st.info("Silakan upload atau tautkan ZIP data untuk mulai.")
