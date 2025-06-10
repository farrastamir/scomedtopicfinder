import streamlit as st
import pandas as pd
import zipfile
import urllib.request
import re
from collections import Counter

# ==================================================
# ⚙️  BASIC CONFIGURATION & PAGE HEADER
# ==================================================

st.set_page_config(layout="wide")
st.title("📰 Topic Summary NoLimit Dashboard — ONM & Sosmed")

# ==================================================
# 📦  DATA INGESTION LAYER (memory‑safe)
# ==================================================

@st.cache_data(show_spinner=False)
def extract_csv_from_zip(zip_file):
    """Read every *.csv in the ZIP into a single string‑dtype DataFrame."""
    try:
        with zipfile.ZipFile(zip_file, "r") as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_files:
                st.error("❌ Tidak ada file .csv di dalam ZIP.")
                return pd.DataFrame()

            dfs = []
            for fn in csv_files:
                try:
                    with z.open(fn) as fh:
                        part = pd.read_csv(
                            fh,
                            delimiter=";",
                            quotechar='"',
                            engine="python",
                            on_bad_lines="skip",
                            dtype=str,           # keep columns as str → lighter & predictable
                        )
                        dfs.append(part)
                except Exception as e:
                    st.warning(f"⚠️  Gagal membaca {fn}: {e}")
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    except zipfile.BadZipFile:
        st.error("❌ File ZIP tidak valid.")
        return pd.DataFrame()

# Helper: drop unused columns very early

def trim_columns(df: pd.DataFrame, needed_cols: set):
    present = [c for c in df.columns if c.lower() in needed_cols]
    missing = needed_cols - {c.lower() for c in present}
    for col in missing:  # create empty so downstream code always finds the column
        df[col] = ""
    return df[present + list(missing)]

# ==================================================
# 🔍  ADVANCED KEYWORD PARSER & MATCHER
# ==================================================

def parse_advanced_keywords(query: str):
    query = query.strip()
    if not query:
        return [], [], []

    include_groups, exact_phrases, exclude_words = [], [], []
    token_pat = r'"[^"]+"|\([^)]+\)|\S+'
    tokens = re.findall(token_pat, query)
    for tok in tokens:
        if tok.startswith('"') and tok.endswith('"'):
            exact_phrases.append(tok.strip('"'))
        elif tok.startswith('-'):
            exclude_words.extend(tok[1:].strip('()').split())
        elif tok.startswith('(') and tok.endswith(')'):
            include_groups.append([w.strip() for w in tok.strip('()').split('OR') if w.strip()])
        else:
            include_groups.append([tok.strip()])
    return include_groups, exact_phrases, exclude_words

def match_advanced(text: str, includes, phrases, excludes):
    low = text.lower()
    if any(w.lower() in low for w in excludes):
        return False
    if any(p.lower() not in low for p in phrases):
        return False
    for grp in includes:
        if not any(w.lower() in low for w in grp):
            return False
    return True

# ==================================================
# ☁️  WORD‑CLOUD SUPPORT FUNCTIONS
# ==================================================

@st.cache_data(show_spinner=False)
def load_stopwords():
    url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt"
    try:
        return set(pd.read_csv(url, header=None)[0].tolist())
    except Exception:
        return set()

STOPWORDS = load_stopwords()

def build_word_freq(text_series: pd.Series, max_words=500):
    tokens = re.findall(r"\b\w{3,}\b", " ".join(text_series.tolist()).lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    return Counter(tokens).most_common(max_words)

# ==================================================
# 🎨  HELPER FOR COLOURED SENTIMENT TAGS
# ==================================================

def sentiment_badge(val):
    s = str(val).lower()
    colors = {"positive": "green", "negative": "red", "neutral": "gray"}
    if s in colors:
        return f'<span style="color:{colors[s]};font-weight:bold">{s}</span>'
    return s

# ==================================================
# 📊  ONLINE NEWS DASHBOARD (ONM)
# ==================================================

def run_onm_dashboard(df: pd.DataFrame):
    # ↓ only keep used columns for performance
    df = trim_columns(
        df,
        {"title", "body", "url", "tier", "sentiment", "label"},
    )

    for c in ["title", "body", "url", "sentiment"]:
        df[c] = df[c].astype(str).str.strip("'")
    df["label"].fillna("", inplace=True)
    df["tier"].fillna("-", inplace=True)

    # ───────── Sidebar filters & STATS ─────────
    st.sidebar.markdown("## 🎛️ Filter — ONM")
    stats_box = st.sidebar.empty()

    sentiments_all = sorted(df["sentiment"].str.lower().unique())
    sentiment_filter = st.sidebar.selectbox("Sentimen", ["All"] + sentiments_all)

    all_labels = sorted({l.strip() for sub in df["label"] for l in str(sub).split(',') if l.strip()})
    label_filter = st.sidebar.selectbox("Label", ["All"] + all_labels)

    if st.sidebar.button("🔄 Clear", key="onm_clear"):
        st.session_state.update({"onm_kw": "", "onm_hl": ""})
    kw_input = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="onm_kw")
    hl_input = st.sidebar.text_input("Highlight Kata", key="onm_hl")

    # ───────── Filtering ─────────
    mask = pd.Series(True, index=df.index)
    if sentiment_filter != "All":
        mask &= df["sentiment"].str.lower() == sentiment_filter
    if label_filter != "All":
        mask &= df["label"].apply(lambda x: label_filter in [s.strip() for s in str(x).split(',')])

    inc, phr, exc = parse_advanced_keywords(kw_input)
    if kw_input:
        kw_mask = df["title"].apply(lambda x: match_advanced(x, inc, phr, exc)) | df["body"].apply(lambda x: match_advanced(x, inc, phr, exc))
        mask &= kw_mask

    filt = df[mask]

    # ───────── STATISTICS box ─────────
    s_low = filt["sentiment"].str.lower()
    stats_html = f"""
    <div style='text-align:center;font-weight:bold;font-size:20px;'>📊 Statistik</div>
    <div style='font-size:30px;font-weight:bold;margin-top:2px'>{len(filt):,}</div>
    <div style='margin-top:2px;font-size:15px;'>
        <span style='color:green;'>🟢 {(s_low=='positive').sum()}</span>&nbsp;|&nbsp;
        <span style='color:gray;'>⚪ {(s_low=='neutral').sum()}</span>&nbsp;|&nbsp;
        <span style='color:red;'>🔴 {(s_low=='negative').sum()}</span>
    </div>"""
    stats_box.markdown(stats_html, unsafe_allow_html=True)

    # ───────── Aggregation ─────────
    def first_url(subdf):
        return subdf["url"].iloc[0] if not subdf["url"].empty else "-"

    grouped = (
        filt.groupby("title", sort=False)
        .agg(
            Total=("title", "size"),
            Sentiment=("sentiment", lambda x: x.mode().iloc[0] if not x.mode().empty else "-"),
            LinkSample=("title", lambda x: first_url(filt[filt["title"] == x.iloc[0]])),
        )
        .sort_values("Total", ascending=False)
        .reset_index()
    )

    # Formatting
    grouped["Sentiment"] = grouped["Sentiment"].apply(sentiment_badge)
    grouped["LinkSample"] = grouped["LinkSample"].apply(lambda u: f'<a href="{u}" target="_blank">Link</a>' if u and u != "-" else "-")

    hl_words = {w.strip('"').lower() for w in re.findall(r'"[^"]+"|\S+', hl_input)}
    def highlight(text):
        for w in hl_words:
            text = re.sub(f"(?i)({re.escape(w)})", r"<mark>\1</mark>", text)
        return text
    grouped["title"] = grouped["title"].apply(highlight)

    # ───────── Display ─────────
    col1, col2 = st.columns([0.72, 0.28])
    with col1:
        st.markdown("### 📊 Ringkasan Topik (ONM)")
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(grouped.to_html(escape=False, index=False), unsafe_allow_html=True)
    with col2:
        if st.checkbox("WordCloud", key="wc_onm"):
            wc_df = pd.DataFrame(build_word_freq(filt["title"] + " " + filt["body"]), columns=["Kata", "Jumlah"])
            st.dataframe(wc_df, use_container_width=True)

# ==================================================
# 📊  SOCIAL MEDIA DASHBOARD
# ==================================================

def run_sosmed_dashboard(df: pd.DataFrame):
    needed = {
        "content", "post_type", "final_sentiment", "label", "specific_resource",
        "object_group", "reply_to_original_id", "original_id", "link"
    }
    df = trim_columns(df, needed)
    df.columns = [c.lower() for c in df.columns]

    df["label"].fillna("", inplace=True)
    df["content"] = df["content"].astype(str)

    st.sidebar.markdown("## 🎛️ Filter — Sosmed")
    stats_box = st.sidebar.empty()

    # Filters
    scope = st.sidebar.radio("Conversation", ("All", "Tanpa Comment", "Tanpa Post"), index=0)

    platforms = sorted({p for p in df["specific_resource"].dropna() if str(p).strip()})
    selected_plat = st.sidebar.multiselect("Platform", platforms, default=platforms) if platforms else []

    sentiments_all = sorted(df["final_sentiment"].dropna().str.lower().unique())
    sent_sel = st.sidebar.selectbox("Sentimen", ["All"] + sentiments_all)

    all_labels = sorted({l.strip() for sub in df["label"] for l in str(sub).split(',') if l.strip()})
    lbl_sel = st.sidebar.selectbox("Label", ["All"] + all_labels)

    groups = sorted({g for g in df["object_group"].dropna() if str(g).strip()})
    grp_sel = st.sidebar.selectbox("Group", ["All"] + groups) if groups else "All"

    if st.sidebar.button("🔄 Clear", key="soc_clear"):
        st.session_state.update({"soc_kw": "", "soc_hl": "", "soc_wc_dyn": True})
    kw_input = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="soc_kw")
    hl_input = st.sidebar.text_input("Highlight Kata", key="soc_hl")

    show_wc = st.sidebar.checkbox("WordCloud", key="soc_wc", value=True)
    dyn_wc = st.sidebar.checkbox("Dinamis", key="soc_wc_dyn", value=True)

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if selected_plat:
        mask &= df["specific_resource"].isin(selected_plat)
    if sent_sel != "All":
        mask &= df["final_sentiment"].str.lower() == sent_sel
    if lbl_sel != "All":
        mask &= df["label"].apply(lambda x: lbl_sel in [s.strip() for s in str(x).split(',')])
    if grp_sel != "All":
        mask &= df["object_group"] == grp_sel

    inc, phr, exc = parse_advanced_keywords(kw_input)
    if kw_input:
        mask &= df["content"].apply(lambda x: match_advanced(x, inc, phr, exc))

    filt = df[mask]

    # Statistics box
    s_low = filt["final_sentiment"].str.lower()
    stats_html = f"""
    <div style='text-align:center;font-weight:bold;font-size:20px;'>📊 Statistik</div>
    <div style='font-size:30px;font-weight:bold;margin-top:2px'>{len(filt):,}</div>
    <div style='margin-top:2px;font-size:15px;'>
        <span style='color:green;'>🟢 {(s_low=='positive').sum()}</span>&nbsp;|&nbsp;
        <span style='color:gray;'>⚪ {(s_low=='neutral').sum()}</span>&nbsp;|&nbsp;
        <span style='color:red;'>🔴 {(s_low=='negative').sum()}</span>
    </div>"""
    stats_box.markdown(stats_html, unsafe_allow_html=True)

    # Scope separation
    is_comment = filt["reply_to_original_id"].notna()
    is_post = filt["post_type"].str.lower() == "post_made"

    if scope == "Tanpa Comment":
        filt_scope = filt[~is_comment]
    elif scope == "Tanpa Post":
        filt_scope = filt[~is_post]
    else:
        filt_scope = filt

    # Value & sentiment calculation
    comments = filt_scope[is_comment]
    com_count = comments.groupby("reply_to_original_id", sort=False).size()
    com_sent_mode = comments.groupby("reply_to_original_id")["final_sentiment"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    posts = filt_scope[is_post].copy()
    posts["value"] = posts["original_id"].map(com_count).fillna(0).astype(int)
    posts["sent_final"] = posts["original_id"].map(com_sent_mode).fillna(posts["final_sentiment"])

    talk = filt_scope[~is_post].copy()
    talk["value"] = 1
    talk["sent_final"] = talk["final_sentiment"]

    if scope == "Tanpa Post":
        base = talk
    elif scope == "Tanpa Comment":
        base = pd.concat([talk, posts], ignore_index=True)
    else:
        base = pd.concat([filt_scope.assign(value=1), posts], ignore_index=True)

    # Summary
    def first_link(subdf):
        val = subdf["link"].iloc[0] if "link" in subdf else ""
        return val if val else "-"

    summary = (
        base.groupby("content", sort=False)
        .agg(
            Total=("value", "sum"),
            Sentiment=("sent_final", lambda x: x.mode().iloc[0] if not x.mode().empty else "-"),
            Platform=("specific_resource", lambda x: ','.join(sorted({p for p in x.dropna() if str(p).strip()}))),
            Label=("label", lambda x: ','.join(sorted({l.strip() for s in x for l in str(s).split(',') if l.strip()}))),
            Group=("object_group", lambda x: ','.join(sorted({g for g in x.dropna() if str(g).strip()}))),
            LinkSample=("link", lambda x: first_link(x.to_frame()))
        )
        .sort_values("Total", ascending=False)
        .reset_index()
    )

    summary["Sentiment"] = summary["Sentiment"].apply(sentiment_badge)
    summary["LinkSample"] = summary["LinkSample"].apply(lambda u: f'<a href="{u}" target="_blank">Link</a>' if u and u != "-" else "-")

    hl_words = {w.strip('"').lower() for w in re.findall(r'"[^"]+"|\S+', hl_input)}
    def highl(text):
        for w in hl_words:
            text = re.sub(f"(?i)({re.escape(w)})", r"<mark>\1</mark>", text)
        return text
    summary["content"] = summary["content"].apply(highl)

    # Display
    c1, c2 = st.columns([0.72, 0.28])
    with c1:
        st.markdown("### 📊 Ringkasan Topik (Sosmed)")
        st.caption(f"Dataset: {len(filt):,} row | Summary: {len(summary):,} baris | Mode: {scope}")
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(summary.to_html(escape=False, index=False), unsafe_allow_html=True)
    with c2:
        if show_wc:
            wc_source = base if dyn_wc else df
            wc_df = pd.DataFrame(build_word_freq(wc_source["content"], 500), columns=["Kata", "Jumlah"])
            st.markdown("### ☁️ Word Cloud (Top 500)")
            st.dataframe(wc_df, use_container_width=True)

# ==================================================
# 🚀  ENTRY POINT
# ==================================================

st.markdown("### 📁 Pilih sumber data")
input_mode = st.radio("Input ZIP via:", ["Upload File", "Link Download"], horizontal=True)

zip_data = None
if input_mode == "Upload File":
    up = st.file_uploader("Unggah file ZIP", type="zip")
    if up:
        zip_data = up
else:
    url = st.text_input("Masukkan URL file ZIP")
    if st.button("Proceed") and url:
        try:
            tmp_path = "/tmp/data.zip"
            urllib.request.urlretrieve(url, tmp_path)
            zip_data = tmp_path
        except Exception as e:
            st.error(f"❌ Gagal mengunduh: {e}")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = None

if zip_data:
    with st.spinner("📖 Memproses data …"):
        df_all = extract_csv_from_zip(zip_data)
        if df_all.empty:
            st.error("❌ Tidak ada data dapat dibaca.")
        else:
            st.session_state["last_df"] = df_all.copy()

if st.session_state["last_df"] is not None:
    data = st.session_state["last_df"]
    cols_lc = [c.lower() for c in data.columns]
    if "tier" in cols_lc:  # ONM route
        run_onm_dashboard(data.copy())
    else:                  # Sosmed route
        required = {"content", "post_type", "final_sentiment"}
        if not required.issubset(cols_lc):
            st.error("❌ Kolom wajib sosmed tidak lengkap: " + ", ".join(sorted(required)))
        else:
            run_sosmed_dashboard(data.copy())
else:
    st.info("Silakan upload atau masukkan URL ZIP untuk mulai.")
