import streamlit as st
import pandas as pd
import zipfile
import urllib.request
import re
from collections import Counter

# ======================================
# ⚙️  BASIC CONFIGURATION & PAGE HEADER
# ======================================

st.set_page_config(layout="wide")
st.title("📰 Topic Summary NoLimit Dashboard — ONM & Sosmed")


# ======================================
# 📦  DATA INGESTION LAYER
# ======================================

@st.cache_data(show_spinner=False)
def extract_csv_from_zip(zip_file):
    """Extract all *.csv inside a ZIP archive and concatenate them into a single DataFrame."""
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
                            on_bad_lines="skip",
                            engine="python",
                        )
                        dfs.append(part)
                except Exception as e:
                    st.warning(f"⚠️  Gagal membaca {fn}: {e}")
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    except zipfile.BadZipFile:
        st.error("❌ File yang di‑upload bukan ZIP yang valid.")
        return pd.DataFrame()


# ======================================
# 🔍  ADVANCED KEYWORD PARSER & MATCHER
# ======================================

def parse_advanced_keywords(query: str):
    """Return include groups, exact phrases and exclude list from advanced search string."""
    query = query.strip()
    if not query:
        return [], [], []

    include_groups, exact_phrases, exclude_words = [], [], []
    token_pat = r'"[^"]+"|\([^)]+\)|\S+'
    for tok in re.findall(token_pat, query):
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
    if any(w.lower() in low for w in excludes):        # ❶ Exclude hits first
        return False
    if any(p.lower() not in low for p in phrases):     # ❷ All phrases must exist
        return False
    for grp in includes:                               # ❸ At least one term per OR‑group
        if not any(w.lower() in low for w in grp):
            return False
    return True


# ======================================
# ☁️  WORD‑CLOUD SUPPORT FUNCTIONS
# ======================================

@st.cache_data(show_spinner=False)
def load_stopwords():
    url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt"
    try:
        return set(pd.read_csv(url, header=None)[0].tolist())
    except Exception:
        return set()

STOPWORDS = load_stopwords()


def build_word_freq(text_series: pd.Series, max_words: int = 500):
    tokens = re.findall(r"\b\w{3,}\b", " ".join(text_series.tolist()).lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    return Counter(tokens).most_common(max_words)


# ======================================
# 🎨  HELPER FOR COLOURED SENTIMENT TAGS
# ======================================

def sentiment_badge(s):
    s = str(s).lower()
    col = {"positive": "green", "negative": "red", "neutral": "gray"}.get(s)
    return f'<span style="color:{col};font-weight:bold">{s}</span>' if col else s


# ======================================
# 📊  ONM (ONLINE NEWS) DASHBOARD
# ======================================

def run_onm_dashboard(df: pd.DataFrame):
    # Basic cleaning
    for c in ["title", "body", "url", "sentiment"]:
        df[c] = df[c].astype(str).str.strip("'")

    df["label"].fillna("", inplace=True)
    df["tier"].fillna("-", inplace=True)
    df["tier"] = pd.Categorical(df["tier"], ["Tier 1", "Tier 2", "Tier 3", "-", ""], True)

    st.sidebar.markdown("## 🎛️ Filter — ONM")

    # Sidebar filters
    sentiments_all = sorted(df["sentiment"].str.lower().unique())
    sentiment_filter = st.sidebar.selectbox("Sentimen", ["All"] + sentiments_all)

    all_labels = sorted({l.strip() for sub in df["label"] for l in str(sub).split(',') if l.strip()})
    label_filter = st.sidebar.selectbox("Label", ["All"] + all_labels)

    if st.sidebar.button("🔄 Clear Filter", key="clr_onm"):
        st.session_state.update({"onm_kw": "", "onm_hl": ""})

    kw_input = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="onm_kw")
    hl_input = st.sidebar.text_input("Highlight Kata", key="onm_hl")

    # Apply filters
    filt = df.copy()
    if sentiment_filter != "All":
        filt = filt[filt["sentiment"].str.lower() == sentiment_filter]
    if label_filter != "All":
        filt = filt[filt["label"].apply(lambda x: label_filter in [l.strip() for l in str(x).split(',')])]

    inc, phr, exc = parse_advanced_keywords(kw_input)
    if kw_input:
        mask = filt["title"].apply(lambda x: match_advanced(x, inc, phr, exc)) | filt["body"].apply(lambda x: match_advanced(x, inc, phr, exc))
        filt = filt[mask]

    # Aggregate per title
    def best_link(sub_df):
        for t in ["Tier 1", "Tier 2", "Tier 3", "-", ""]:
            sel = sub_df[sub_df["tier"] == t]["url"]
            if not sel.empty:
                return sel.iloc[0]
        return "-"

    grouped = (
        filt.groupby("title")
        .agg(
            Jumlah=("title", "count"),
            Sentimen=("sentiment", lambda x: x.mode().iloc[0] if not x.mode().empty else "-"),
            Link=("title", lambda x: best_link(filt[filt["title"] == x.iloc[0]])),
        )
        .reset_index()
        .sort_values("Jumlah", ascending=False)
    )
    grouped["Sentimen"] = grouped["Sentimen"].apply(sentiment_badge)

    # Highlight
    hl_set = {h.strip('"').lower() for h in re.findall(r'"[^"]+"|\S+', hl_input)}
    def highlight(text):
        for w in hl_set:
            text = re.sub(f"(?i)({re.escape(w)})", r"<mark>\1</mark>", text)
        return text
    grouped["title"] = grouped["title"].apply(highlight)

    # Display
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown("### 📊 Ringkasan Topik (ONM)")
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(grouped.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        if st.checkbox("Tampilkan WordCloud", key="wc_onm"):
            wc_df = pd.DataFrame(build_word_freq(filt["title"] + " " + filt["body"]), columns=["Kata", "Jumlah"])
            st.dataframe(wc_df, use_container_width=True)


# ======================================
# 📊  SOCIAL MEDIA DASHBOARD
# ======================================

def run_sosmed_dashboard(df: pd.DataFrame):
    """Dashboard for social‑media datasets. Required cols: content, post_type, final_sentiment."""

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Ensure presence of optional columns
    optional_defaults = {
        "specific_resource": "",
        "label": "",
        "object_group": "",
        "reply_to_original_id": pd.NA,
        "original_id": pd.NA,
    }
    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    # Basic cleanup
    df["content"] = df["content"].astype(str)
    df["label"].fillna("", inplace=True)

    st.sidebar.markdown("## 🎛️ Filter — Sosmed")

    # 1️⃣ Conversation scope
    conv_scope = st.sidebar.radio("Conversation Scope", ("All", "Tanpa Comment", "Tanpa Post"), index=0)

    # 2️⃣ Platform filter (if column exists)
    platforms = sorted([p for p in df["specific_resource"].dropna().unique() if str(p).strip()])
    platform_sel = st.sidebar.multiselect("Platform", platforms, default=platforms) if platforms else []

    # 3️⃣ Sentiment
    sentiments_all = sorted(df["final_sentiment"].dropna().str.lower().unique())
    sent_sel = st.sidebar.selectbox("Sentimen", ["All"] + sentiments_all)

    # 4️⃣ Label
    all_labels = sorted({l.strip() for sub in df["label"] for l in str(sub).split(',') if l.strip()})
    lbl_sel = st.sidebar.selectbox("Label", ["All"] + all_labels)

    # 5️⃣ Group
    groups = sorted([g for g in df["object_group"].dropna().unique() if str(g).strip()])
    grp_sel = st.sidebar.selectbox("Group (object_group)", ["All"] + groups) if groups else "All"

    # 6️⃣ Keywords / highlight
    if st.sidebar.button("🔄 Clear Filter", key="clr_soc"):
        st.session_state.update({"soc_kw": "", "soc_hl": "", "soc_wc_dyn": True})
    kw_input = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="soc_kw")
    hl_input = st.sidebar.text_input("Highlight Kata", key="soc_hl")

    show_wc = st.sidebar.checkbox("Tampilkan WordCloud", key="soc_wc_show", value=True)
    dyn_wc = st.sidebar.checkbox("Word Cloud Dinamis", key="soc_wc_dyn", value=True)

    # ------- APPLY FILTERS -------
    filt = df.copy()
    if platform_sel:
        filt = filt[filt["specific_resource"].isin(platform_sel)]
    if sent_sel != "All":
        filt = filt[filt["final_sentiment"].str.lower() == sent_sel]
    if lbl_sel != "All":
        filt = filt[filt["label"].apply(lambda x: lbl_sel in [l.strip() for l in str(x).split(',')])]
    if grp_sel != "All":
        filt = filt[filt["object_group"] == grp_sel]

    inc, phr, exc = parse_advanced_keywords(kw_input)
    if kw_input:
        filt = filt[filt["content"].apply(lambda x: match_advanced(x, inc, phr, exc))]

    # Identify comment / post rows
    is_comment = filt["reply_to_original_id"].notna()
    is_post = filt["post_type"].str.lower() == "post_made"

    if conv_scope == "Tanpa Comment":
        filt = filt[~is_comment]
    elif conv_scope == "Tanpa Post":
        filt = filt[~is_post]

    # ------- VALUE & SENTIMENT CALCULATION -------
    comments = filt[is_comment]
    comment_cnt = comments.groupby("reply_to_original_id").size()
    comment_sent_mode = comments.groupby("reply_to_original_id")["final_sentiment"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    posts = filt[is_post].copy()
    posts["value"] = posts["original_id"].map(comment_cnt).fillna(0).astype(int)
    posts["sentiment_final"] = posts["original_id"].map(comment_sent_mode).fillna(posts["final_sentiment"])

    talk = filt[~is_post].copy()
    talk["value"] = 1
    talk["sentiment_final"] = talk["final_sentiment"]

    if conv_scope == "Tanpa Post":
        base = talk
    elif conv_scope == "Tanpa Comment":
        base = pd.concat([talk, posts], ignore_index=True)
    else:
        base = pd.concat([filt.assign(value=1), posts], ignore_index=True)

    # ------- SUMMARY TABLE -------
    summary = (
        base.groupby("content")
        .agg(
            Total=("value", "sum"),
            Sentiment=("sentiment_final", lambda x: x.mode().iloc[0] if not x.mode().empty else "-"),
            Platform=("specific_resource", lambda x: ",".join(sorted({p for p in x.dropna() if str(p).strip()}))),
            Label=("label", lambda x: ",".join(sorted({l.strip() for s in x for l in str(s).split(',') if l.strip()}))),
            Group=("object_group", lambda x: ",".join(sorted({g for g in x.dropna() if str(g).strip()}))),
        )
        .reset_index()
        .sort_values("Total", ascending=False)
    )

    # Highlight
    hl_set = {h.strip('"').lower() for h in re.findall(r'"[^"]+"|\S+', hl_input)}
    def high(text):
        for w in hl_set:
            text = re.sub(f"(?i)({re.escape(w)})", r"<mark>\1</mark>", text)
        return text
    summary["content"] = summary["content"].apply(high)
    summary["Sentiment"] = summary["Sentiment"].apply(sentiment_badge)

    # ------- DISPLAY -------
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown("### 📊 Ringkasan Topik (Sosmed)")
        st.caption(f"Dataset: {len(filt):,} row | Summary: {len(summary):,} baris | Mode: {conv_scope}")
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(summary.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        if show_wc:
            wc_source = base if dyn_wc else df
            wc_df = pd.DataFrame(build_word_freq(wc_source["content"], 500), columns=["Kata", "Jumlah"])
            st.markdown("### ☁️  Word Cloud (Top 500)")
            st.dataframe(wc_df, use_container_width=True)


# ======================================
# 🚀  APP ENTRY POINT
# ======================================

st.markdown("### 📁 Pilih sumber data")
input_mode = st.radio("Input ZIP via:", ["Upload File", "Link Download"], horizontal=True)

zip_data = None
if input_mode == "Upload File":
    upfile = st.file_uploader("Unggah file ZIP", type="zip")
    if upfile:
        zip_data = upfile
else:
    zip_url = st.text_input("Masukkan URL file ZIP")
    if st.button("Proceed") and zip_url:
        try:
            temp_path = "/tmp/download.zip"
            urllib.request.urlretrieve(zip_url, temp_path)
            zip_data = temp_path
        except Exception as e:
            st.error(f"❌ Gagal mengunduh: {e}")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = None

if zip_data:
    with st.spinner("📖 Membaca dan memproses data ..."):
        df_all = extract_csv_from_zip(zip_data)
        if df_all.empty:
            st.error("❌ DataFrame kosong setelah ekstraksi.")
        else:
            st.session_state["last_df"] = df_all.copy()

if st.session_state["last_df"] is not None:
    df_loaded = st.session_state["last_df"]
    cols_lower = [c.lower() for c in df_loaded.columns]
    if "tier" in cols_lower:
        run_onm_dashboard(df_loaded.copy())
    else:
        required = {"content", "post_type", "final_sentiment"}
        if not required.issubset(set(cols_lower)):
            st.error("❌ Dataset sosmed tidak memiliki kolom wajib: " + ", ".join(sorted(required)))
        else:
            run_sosmed_dashboard(df_loaded.copy())
else:
    st.info("Silakan upload atau unduh ZIP untuk melihat ringkasan topik.")
