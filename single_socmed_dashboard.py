import streamlit as st
import pandas as pd
import zipfile
import urllib.request
import re
import textwrap
from collections import Counter

# ----------------- CONFIG -----------------
st.set_page_config(layout="wide")
st.title("📰 Topic Summary NoLimit Dashboard — ONM & Sosmed")

# ----------------- ZIP LOADER -----------------
@st.cache_data(show_spinner=False)
def extract_csv_from_zip(zf):
    """Return a single DataFrame (all‑string columns) from every *.csv inside a ZIP."""
    try:
        with zipfile.ZipFile(zf, "r") as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_files:
                st.error("❌ ZIP tidak berisi file .csv.")
                return pd.DataFrame()

            dfs = []
            for fn in csv_files:
                with z.open(fn) as fh:
                    dfs.append(pd.read_csv(
                        fh, delimiter=";", quotechar='"',
                        engine="python", on_bad_lines="skip", dtype=str))
            return pd.concat(dfs, ignore_index=True)
    except zipfile.BadZipFile:
        st.error("❌ ZIP tidak valid.")
        return pd.DataFrame()

def trim_columns(df: pd.DataFrame, cols_needed: set):
    keep = [c for c in df.columns if c.lower() in cols_needed]
    missing = cols_needed - {c.lower() for c in keep}
    for m in missing:
        df[m] = ""
    return df[keep + list(missing)]

# ----------------- KEYWORD UTILS -----------------
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

def match_advanced(text: str, inc, phr, exc):
    low = text.lower()
    if any(w in low for w in exc):
        return False
    if any(p not in low for p in phr):
        return False
    return all(any(w in low for w in g) for g in inc)

# ----------------- WORD-CLOUD UTILS -----------------
@st.cache_data(show_spinner=False)
def load_stopwords():
    url = ("https://raw.githubusercontent.com/stopwords-iso/"
           "stopwords-id/master/stopwords-id.txt")
    try:
        return set(pd.read_csv(url, header=None)[0].tolist())
    except Exception:
        return set()

STOPWORDS = load_stopwords()

def word_freq(series: pd.Series, top_n=500):
    tokens = re.findall(r"\b\w{3,}\b", " ".join(series).lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    return Counter(tokens).most_common(top_n)

# ----------------- SENTIMENT BADGE -----------------
def badge(val):
    s = str(val).lower()
    colors = {"positive": "green", "negative": "red", "neutral": "gray"}
    return f'<span style="color:{colors.get(s, "black")};font-weight:bold">{s}</span>'

# ----------------- ONM DASHBOARD -----------------
def run_onm(df: pd.DataFrame):
    df = trim_columns(df, {"title", "body", "url", "tier", "sentiment", "label"})
    for col in ["title", "body", "url", "sentiment"]:
        df[col] = df[col].astype(str).str.strip("'")
    df["tier"].fillna("-", inplace=True)
    df["label"].fillna("", inplace=True)

    st.sidebar.markdown("## 🎛️ Filter — ONM")
    stat_box = st.sidebar.empty()

    sent_sel = st.sidebar.selectbox(
        "Sentimen", ["All"] + sorted(df["sentiment"].str.lower().unique()))
    lab_sel = st.sidebar.selectbox(
        "Label", ["All"] + sorted({l.strip() for sub in df["label"]
                                   for l in str(sub).split(',') if l.strip()}))

    if st.sidebar.button("🔄 Clear", key="onm_clear"):
        st.session_state.update(onm_kw="", onm_hl="")
    kw = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="onm_kw")
    hl = st.sidebar.text_input("Highlight Kata", key="onm_hl")

    # Filter rows
    mask = pd.Series(True, index=df.index)
    if sent_sel != "All":
        mask &= df["sentiment"].str.lower() == sent_sel
    if lab_sel != "All":
        mask &= df["label"].apply(
            lambda x: lab_sel in [s.strip() for s in str(x).split(',')])
    inc, phr, exc = parse_advanced_keywords(kw)
    if kw:
        mask &= (df["title"].apply(lambda x: match_advanced(x, inc, phr, exc)) |
                 df["body"].apply(lambda x: match_advanced(x, inc, phr, exc)))

    filt = df[mask]

    # Sidebar statistic
    sl = filt["sentiment"].str.lower()
    stat_box.markdown(
        f"<div style='text-align:center;font-weight:bold;font-size:20px;'>📊 Statistik</div>"
        f"<div style='font-size:28px;font-weight:bold'>Article: {len(filt):,}</div>"
        f"<div style='font-size:15px;'>"
        f"<span style='color:green;'>🟢 {(sl=='positive').sum()}</span> | "
        f"<span style='color:gray;'>⚪ {(sl=='neutral').sum()}</span> | "
        f"<span style='color:red;'>🔴 {(sl=='negative').sum()}</span></div>",
        unsafe_allow_html=True)

    # Link priority
    rank = {"tier 1": 0, "tier 2": 1, "tier 3": 2, "-": 3, "": 4}
    def best_link(g):
        g["rk"] = g["tier"].str.lower().map(rank).fillna(4)
        g = g.sort_values("rk")
        return g["url"].iloc[0] if not g["url"].empty else "-"

    grouped = (filt.groupby("title", sort=False)
               .apply(best_link).to_frame("Link")
               .join(filt.groupby("title").agg(
                     Total=("title", "size"),
                     Sentiment=("sentiment",
                                lambda x: x.mode().iloc[0] if not x.mode().empty else "-")),
                     how="left")
               .reset_index().sort_values("Total", ascending=False))

    grouped["Sentiment"] = grouped["Sentiment"].apply(badge)
    grouped["Link"] = grouped["Link"].apply(
        lambda u: f'<a href="{u}" target="_blank">Link</a>' if u and u != "-" else "-")

    # Highlight
    hl_set = {h.strip('"').lower() for h in re.findall(r'"[^"]+"|\S+', hl)}
    if hl_set:
        pattern = re.compile(f"(?i)({'|'.join(map(re.escape, hl_set))})")
        grouped["title"] = grouped["title"].apply(
            lambda t: pattern.sub(r"<mark>\1</mark>", t))

    # Show table + word cloud
    col1, col2 = st.columns([0.72, 0.28])
    with col1:
        st.markdown("### 📊 Ringkasan Topik (ONM)")
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(grouped.to_html(escape=False, index=False),
                 unsafe_allow_html=True)
    with col2:
        if st.checkbox("WordCloud", key="wc_onm"):
            wc_df = pd.DataFrame(
                word_freq(filt["title"] + " " + filt["body"]),
                columns=["Kata", "Jumlah"])
            st.dataframe(wc_df, use_container_width=True)

# ----------------- SOSMED DASHBOARD -----------------
def run_sosmed(df: pd.DataFrame):
    need = {"content", "post_type", "final_sentiment", "specific_resource",
            "reply_to_original_id", "original_id", "link"}
    df = trim_columns(df, need)
    df.columns = [c.lower() for c in df.columns]
    df["content"] = df["content"].astype(str)
    df["final_sentiment"] = df["final_sentiment"].astype(str).str.strip("\"' ")

    st.sidebar.markdown("## 🎛️ Filter — Sosmed")
    stat_box = st.sidebar.empty()

    scope = st.sidebar.radio("Conversation",
                             ("All", "Tanpa Comment", "Tanpa Post"), index=0)
    plats = sorted({p for p in df["specific_resource"].dropna() if str(p).strip()})
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

    mask = pd.Series(True, index=df.index)
    if plat_sel:
        mask &= df["specific_resource"].isin(plat_sel)
    if sent_sel != "All":
        mask &= df["final_sentiment"].str.lower() == sent_sel
    inc, phr, exc = parse_advanced_keywords(kw)
    if kw:
        mask &= df["content"].apply(lambda x: match_advanced(x, inc, phr, exc))
    filt = df[mask]

    # Sidebar stats
    sl = filt["final_sentiment"].str.lower()
    stat_box.markdown(
        f"<div style='text-align:center;font-weight:bold;font-size:20px;'>📊 Statistik</div>"
        f"<div style='font-size:28px;font-weight:bold'>Talk: {len(filt):,}</div>"
        f"<div style='font-size:15px;'>"
        f"<span style='color:green;'>🟢 {(sl=='positive').sum()}</span> | "
        f"<span style='color:gray;'>⚪ {(sl=='neutral').sum()}</span> | "
        f"<span style='color:red;'>🔴 {(sl=='negative').sum()}</span></div>",
        unsafe_allow_html=True)

    # Scope separation
    is_com = filt["reply_to_original_id"].notna()
    is_post = filt["post_type"].str.lower() == "post_made"

    if scope == "Tanpa Comment":
        filt_scope = filt[~is_com]
    elif scope == "Tanpa Post":
        filt_scope = filt[~is_post]
    else:
        filt_scope = filt

    # Engagement
    comments = filt_scope[is_com]
    com_cnt = comments.groupby("reply_to_original_id").size()
    com_sent = comments.groupby("reply_to_original_id")["final_sentiment"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    posts = filt_scope[is_post].copy()
    posts["value"] = posts["original_id"].map(com_cnt).fillna(0).astype(int)
    posts["sent_final"] = posts["original_id"].map(com_sent).fillna(
        posts["final_sentiment"])

    talk = filt_scope[~is_post].copy()
    talk["value"] = 1
    talk["sent_final"] = talk["final_sentiment"]

    if scope == "Tanpa Post":
        base = talk
    elif scope == "Tanpa Comment":
        base = pd.concat([talk, posts], ignore_index=True)
    else:
        # All → include every comment (each counts 1)
        comments_all = filt_scope[is_com].copy()
        comments_all["value"] = 1
        comments_all["sent_final"] = comments_all["final_sentiment"]
        base = pd.concat([talk, posts, comments_all], ignore_index=True)

    # Build summary
    def best_link(series, val_series):
        df_links = pd.DataFrame({"link": series, "val": val_series})
        df_links = df_links.dropna()
        df_links = df_links[df_links["link"].astype(str).str.strip() != ""]
        if df_links.empty:
            return "-"
        return df_links.sort_values("val", ascending=False)["link"].iloc[0]

    def shorten(txt, n=120):
        clean = re.sub(r'<.*?>', '', txt)
        return textwrap.shorten(clean, width=n, placeholder="…")

    summary = (base.groupby("content", sort=False)
               .agg(Total=("value", "sum"),
                    Sentiment=("sent_final",
                               lambda x: x.mode().iloc[0]
                               if not x.mode().empty else "-"),
                    Platform=("specific_resource",
                              lambda x: ', '.join({p.strip()
                                                   for p in x.dropna()
                                                   if p.strip()})),
                    LinkSample=("link", lambda x: best_link(x, base.loc[x.index, "value"])))
               .sort_values("Total", ascending=False)
               .reset_index())

    summary["Sentiment"] = summary["Sentiment"].apply(badge)
    summary["content"] = summary["content"].apply(shorten)
    summary["LinkSample"] = summary["LinkSample"].apply(
        lambda u: f'<a href="{u}" target="_blank">Link</a>' if u and u != "-" else "-")

    # Highlight
    hl_set = {h.strip('"').lower() for h in re.findall(r'"[^"]+"|\S+', hl)}
    if hl_set:
        pattern = re.compile(f"(?i)({'|'.join(map(re.escape, hl_set))})")
        summary["content"] = summary["content"].apply(
            lambda t: pattern.sub(r"<mark>\1</mark>", t))

    # Display
    col1, col2 = st.columns([0.72, 0.28])
    with col1:
        st.markdown("### 📊 Ringkasan Topik (Sosmed)")
        st.caption(f"Dataset: {len(filt):,} | Summary: {len(summary):,}")
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(summary.to_html(escape=False, index=False),
                 unsafe_allow_html=True)
    with col2:
        if show_wc:
            wc_df = pd.DataFrame(
                word_freq(base["content"], 500), columns=["Kata", "Jumlah"])
            st.markdown("### ☁️ Word Cloud (Top 500)")
            st.dataframe(wc_df, use_container_width=True)

# ----------------- ENTRY -----------------
st.markdown("### 📁 Pilih sumber data")
inp_mode = st.radio("Input ZIP via:", ["Upload File", "Link Download"],
                    horizontal=True)

zip_data = None
if inp_mode == "Upload File":
    up = st.file_uploader("Unggah file ZIP", type="zip")
    if up:
        zip_data = up
else:
    url = st.text_input("Masukkan URL ZIP")
    if st.button("Proceed") and url:
        try:
            tmp = "/tmp/data.zip"
            urllib.request.urlretrieve(url, tmp)
            zip_data = tmp
        except Exception as e:
            st.error(f"❌ Gagal mengunduh: {e}")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = None

if zip_data:
    with st.spinner("📖 Memproses data …"):
        df_all = extract_csv_from_zip(zip_data)
        if df_all.empty:
            st.error("❌ DataFrame kosong / gagal dibaca.")
        else:
            st.session_state["last_df"] = df_all.copy()

if st.session_state["last_df"] is not None:
    df_loaded = st.session_state["last_df"]
    cols_lower = [c.lower() for c in df_loaded.columns]

    if "tier" in cols_lower:                       # Assume ONM
        run_onm(df_loaded.copy())
    elif {"content", "post_type", "final_sentiment"}.issubset(cols_lower):
        run_sosmed(df_loaded.copy())                # Assume Sosmed
    else:
        st.error("❌ Struktur kolom tidak cocok format ONM maupun Sosmed.")
else:
    st.info("Unggah atau tautkan ZIP untuk memulai.")
