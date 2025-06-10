import streamlit as st
import pandas as pd
import zipfile
import urllib.request
import re
from collections import Counter

# =============================
# ⚙️  BASIC CONFIGURATION
# =============================

st.set_page_config(layout="wide")
st.title("📰 Topic Summary NoLimit Dashboard — ONM & Sosmed")


# =============================
# 📦  DATA INGESTION LAYER
# =============================

@st.cache_data(show_spinner=False)
def extract_csv_from_zip(zip_file):
    """Extract all *.csv in the uploaded ZIP file and return a concatenated dataframe."""
    with zipfile.ZipFile(zip_file, "r") as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        if not csv_files:
            st.error("❌ Tidak ada file .csv di dalam ZIP.")
            return pd.DataFrame()

        dfs = []
        for f in csv_files:
            try:
                with z.open(f) as fh:
                    df_part = pd.read_csv(
                        fh,
                        delimiter=";",
                        quotechar='"',
                        on_bad_lines="skip",
                        engine="python",
                    )
                    dfs.append(df_part)
            except Exception as e:
                st.warning(f"⚠️  Gagal membaca {f}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# =============================
# 🔍  ADVANCED KEYWORD PARSER
# =============================

def parse_advanced_keywords(query: str):
    """Split the user keyword query into include‑groups, exact phrases, and excluded words."""
    query = query.strip()
    if not query:
        return [], [], []

    include_groups, exact_phrases, exclude_words = [], [], []
    token_pattern = r'"[^"]+"|\([^)]+\)|\S+'
    tokens = re.findall(token_pattern, query)

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
    """Return True if *text* matches the parsed keyword condition sets."""
    low = text.lower()
    if any(w.lower() in low for w in excludes):
        return False
    if any(p.lower() not in low for p in phrases):
        return False
    for group in includes:
        if not any(w.lower() in low for w in group):
            return False
    return True


# =============================
# 🎨  WORD‑CLOUD PREP HELPERS
# =============================

@st.cache_data(show_spinner=False)
def load_stopwords():
    url = (
        "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt"
    )
    try:
        return set(pd.read_csv(url, header=None)[0].tolist())
    except Exception:
        return set()

STOPWORDS = load_stopwords()


def build_word_freq(text_series: pd.Series, max_words: int = 500):
    tokens = re.findall(r"\b\w{3,}\b", " ".join(text_series.tolist()).lower())
    tokens = [t for t in tokens if t not in STOPWORDS]
    return Counter(tokens).most_common(max_words)


# =============================
# 🖼️  SENTIMENT COLOUR TAG
# =============================

def sentiment_badge(s):
    s = str(s).lower()
    if s == "positive":
        color = "green"
    elif s == "negative":
        color = "red"
    elif s == "neutral":
        color = "gray"
    else:
        return s  # unknown label, no styling
    return f'<span style="color:{color};font-weight:bold">{s}</span>'


# =============================
# 📊  ONM (ONLINE NEWS MONITORING) IMPLEMENTATION
# =============================

def run_onm_dashboard(df: pd.DataFrame):
    """Original behaviour — assumed columns: title, body, url, tier, sentiment, label."""
    # --- HOUSEKEEPING
    for col in ["title", "body", "url", "sentiment"]:
        df[col] = df[col].astype(str).str.strip("'")

    df["label"] = df["label"].fillna("")
    df["tier"] = df["tier"].fillna("-")
    df["tier"] = pd.Categorical(
        df["tier"], categories=["Tier 1", "Tier 2", "Tier 3", "-", ""], ordered=True
    )

    # Sidebar filter components
    st.sidebar.markdown("## 🎛️ Filter — ONM")

    # Sentiment
    sentiments_all = sorted(df["sentiment"].str.lower().unique())
    sentiment_filter = st.sidebar.selectbox("Sentimen", options=["All"] + sentiments_all)

    # Label
    all_labels = sorted(
        {
            label.strip()
            for sub in df["label"].astype(str)
            for label in sub.split(",")
            if label.strip()
        }
    )
    label_filter = st.sidebar.selectbox("Label", options=["All"] + all_labels)

    # Keyword, highlight, reset
    if st.sidebar.button("🔄 Clear Filter"):
        st.session_state.update(
            {
                "sentiment_filter": "All",
                "label_filter": "All",
                "keyword_input": "",
                "highlight_words": "",
            }
        )

    keyword_input = st.sidebar.text_input(
        "Kata kunci (\"frasa\" -exclude)", value=st.session_state.get("keyword_input", "")
    )
    highlight_input = st.sidebar.text_input(
        "Highlight Kata", value=st.session_state.get("highlight_words", "")
    )

    # === FILTER LOGIC ===
    filtered = df.copy()
    if sentiment_filter != "All":
        filtered = filtered[filtered["sentiment"].str.lower() == sentiment_filter]
    if label_filter != "All":
        filtered = filtered[
            filtered["label"].apply(
                lambda x: label_filter in [s.strip() for s in str(x).split(",")]
            )
        ]

    # Advanced keyword filter on title & body
    includes, phrases, excludes = parse_advanced_keywords(keyword_input)
    if keyword_input:
        mask_kw = filtered["title"].apply(
            lambda x: match_advanced(x, includes, phrases, excludes)
        ) | filtered["body"].apply(lambda x: match_advanced(x, includes, phrases, excludes))
        filtered = filtered[mask_kw]

    # === SUMMARY TABLE ===
    def best_link(sub):
        for tier in ["Tier 1", "Tier 2", "Tier 3", "-", ""]:
            link = sub[sub["tier"] == tier]["url"]
            if not link.empty:
                return link.iloc[0]
        return "-"

    grouped = (
        filtered.groupby("title")
        .agg(
            Article=("title", "count"),
            Sentiment=("sentiment", lambda x: x.mode().iloc[0] if not x.mode().empty else "-"),
            Link=("title", lambda x: best_link(filtered[filtered["title"] == x.iloc[0]])),
        )
        .reset_index()
        .sort_values(by="Article", ascending=False)
    )

    grouped["Sentiment"] = grouped["Sentiment"].apply(sentiment_badge)

    # === DISPLAY ===
    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        st.markdown("### 📊 Ringkasan Topik (ONM)")
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(grouped.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if st.checkbox("Tampilkan WordCloud", key="wc_onm"):
            wc_data = build_word_freq(
                filtered["title"].astype(str) + " " + filtered["body"].astype(str)
            )
            wc_df = pd.DataFrame(wc_data, columns=["Kata", "Jumlah"])
            st.dataframe(wc_df, use_container_width=True)


# =============================
# 📊  SOCIAL MEDIA IMPLEMENTATION
# =============================

def run_sosmed_dashboard(df: pd.DataFrame):
    """New behaviour — assumed columns: content, post_type, original_id, reply_to_original_id,
    final_sentiment, label, specific_resource, object_group."""

    # Normalise column names (allow any capitalisation)
    df.columns = [c.lower() for c in df.columns]

    # BASIC CLEANUP
    df["content"] = df["content"].astype(str)
    df["label"] = df.get("label", "").fillna("")

    # ===== SIDEBAR — FILTER SECTION =====
    st.sidebar.markdown("## 🎛️ Filter — Sosmed")

    # 1️⃣ Conversation type selector
    conv_option = st.sidebar.radio(
        "Conversation Scope",
        ("All", "Tanpa Comment", "Tanpa Post"),
        index=0,
        help="Pilih cara perhitungan nilai untuk postingan dan comment.",
    )

    # 2️⃣ Platform filter
    platforms = sorted(df["specific_resource"].dropna().unique())
    platform_filter = st.sidebar.multiselect(
        "Platform", options=platforms, default=platforms, help="Saring berdasarkan platform."
    )

    # 3️⃣ Sentiment filter
    sentiments_all = sorted(df["final_sentiment"].dropna().str.lower().unique())
    sentiment_filter = st.sidebar.selectbox("Sentimen", ["All"] + sentiments_all)

    # 4️⃣ Label filter
    all_labels = sorted(
        {
            l.strip()
            for sub in df["label"].astype(str)
            for l in sub.split(",")
            if l.strip()
        }
    )
    label_filter = st.sidebar.selectbox("Label", ["All"] + all_labels)

    # 5️⃣ Group filter
    all_groups = sorted(df["object_group"].dropna().unique())
    group_filter = st.sidebar.selectbox("Group (object_group)", ["All"] + all_groups)

    # 6️⃣ Keywords & highlight
    if st.sidebar.button("🔄 Clear Filter", key="clear_sosmed"):
        st.session_state.update(
            {
                "sos_kw": "",
                "sos_hl": "",
                "sos_dynamic_wc": True,
            }
        )

    kw_input = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="sos_kw")
    hl_input = st.sidebar.text_input("Highlight Kata", key="sos_hl")

    show_wc = st.sidebar.checkbox("Tampilkan WordCloud", key="sos_show_wc", value=True)
    dynamic_wc = st.sidebar.checkbox(
        "Word Cloud Dinamis", key="sos_dynamic_wc", value=True
    )

    # ===== APPLY FILTERS =====
    filt = df.copy()
    if platform_filter:
        filt = filt[filt["specific_resource"].isin(platform_filter)]
    if sentiment_filter != "All":
        filt = filt[filt["final_sentiment"].str.lower() == sentiment_filter]
    if label_filter != "All":
        filt = filt[
            filt["label"].apply(lambda x: label_filter in [s.strip() for s in str(x).split(",")])
        ]
    if group_filter != "All":
        filt = filt[filt["object_group"] == group_filter]

    # Keyword search across content
    inc, phr, exc = parse_advanced_keywords(kw_input)
    if kw_input:
        filt = filt[filt["content"].apply(lambda x: match_advanced(x, inc, phr, exc))]

    # ===== CONVERSATION SCOPE HANDLING =====
    # Identify comment rows (reply_to_original_id has value, regardless of post_type)
    is_comment = filt["reply_to_original_id"].notna()
    is_post = filt["post_type"].str.lower() == "post_made"

    if conv_option == "Tanpa Comment":
        filt = filt[~is_comment]
    elif conv_option == "Tanpa Post":
        filt = filt[~is_post]

    # ===== AGGREGATION LOGIC =====
    # Pre‑compute comment counts & majority sentiment for each original_id
    comments = filt[is_comment]
    comment_counts = comments.groupby("reply_to_original_id").size()
    comment_sent_mode = (
        comments.groupby("reply_to_original_id")["final_sentiment"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )

    # Prepare a dataframe for post rows with derived count & sentiment
    posts = filt[is_post].copy()
    if not posts.empty:
        posts["value"] = posts["original_id"].map(comment_counts).fillna(0).astype(int)
        posts["sentiment_calc"] = posts["original_id"].map(comment_sent_mode)
        posts["sentiment_final"] = posts["sentiment_calc"].fillna(posts["final_sentiment"])
    else:
        posts = pd.DataFrame(columns=filt.columns.tolist() + ["value", "sentiment_final"])

    # TALK / other rows (non‑post)
    talk = filt[~is_post].copy()
    talk["value"] = 1  # each row counts as 1 occurrence
    talk["sentiment_final"] = talk["final_sentiment"]

    # Combine for summary depending on conversation scope
    if conv_option == "Tanpa Post":
        summary_source = talk
    elif conv_option == "Tanpa Comment":
        summary_source = pd.concat([talk, posts], ignore_index=True)
    else:  # All
        summary_source = pd.concat([filt.assign(value=1), posts], ignore_index=True)

    # GROUP BY CONTENT FOR SUMMARY TABLE
    summary = (
        summary_source.groupby("content")
        .agg(
            Total=("value", "sum"),
            Sentiment=("sentiment_final", lambda x: x.mode().iloc[0] if not x.mode().empty else "-"),
            Platform=("specific_resource", lambda x: ",".join(sorted(set(x.dropna())))),
            Label=("label", lambda x: ",".join(sorted({l.strip() for s in x for l in str(s).split(',') if l.strip()}))),
            Group=("object_group", lambda x: ",".join(sorted(set(x.dropna())))),
        )
        .reset_index()
        .sort_values(by="Total", ascending=False)
    )

    # ===== HIGHLIGHT KEYWORDS =====
    highlight_tokens = re.findall(r'"[^"]+"|\S+', hl_input)
    highlight_set = {t.strip('"').lower() for t in highlight_tokens if t}

    def highlighter(text):
        for w in highlight_set:
            text = re.sub(f"(?i)({re.escape(w)})", r"<mark>\1</mark>", text)
        return text

    summary["content"] = summary["content"].apply(highlighter)
    summary["Sentiment"] = summary["Sentiment"].apply(sentiment_badge)

    # ===== DISPLAY =====
    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        st.markdown("### 📊 Ringkasan Topik (Sosmed)")
        st.caption(
            f"Dataset: {len(filt):,} row | Summary: {len(summary):,} baris | Filter mode: {conv_option}"
        )
        st.markdown("<div style='overflow-x:auto;'>", unsafe_allow_html=True)
        st.write(summary.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if show_wc:
            wc_source = summary_source if dynamic_wc else df
            wc_data = build_word_freq(wc_source["content"].astype(str))
            wc_df = pd.DataFrame(wc_data, columns=["Kata", "Jumlah"])
            st.markdown("### ☁️  Word Cloud (Top 500)")
            st.dataframe(wc_df, use_container_width=True)


# =============================
# 🚀  ENTRY POINT
# =============================

st.markdown("### 📁 Pilih sumber data")
input_type = st.radio("Input ZIP via:", ["Upload File", "Link Download"], horizontal=True)

zip_data = None
if input_type == "Upload File":
    up_file = st.file_uploader("Unggah file ZIP", type="zip")
    if up_file:
        zip_data = up_file
else:
    zip_url = st.text_input("Masukkan URL file ZIP")
    if st.button("Proceed") and zip_url:
        try:
            tmp_path = "/tmp/download.zip"
            urllib.request.urlretrieve(zip_url, tmp_path)
            zip_data = tmp_path
        except Exception as e:
            st.error(f"❌ Gagal mengunduh: {e}")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = None

if zip_data:
    with st.spinner("📖 Membaca dan memproses data ..."):
        df_concat = extract_csv_from_zip(zip_data)
        if not df_concat.empty:
            st.session_state["last_df"] = df_concat.copy()
        else:
            st.error("❌ DataFrame kosong setelah ekstraksi.")

if st.session_state["last_df"] is not None:
    df_loaded = st.session_state["last_df"]
    if "tier" in [c.lower() for c in df_loaded.columns]:
        # ONM ROUTE
        run_onm_dashboard(df_loaded.copy())
    else:
        # SOSMED ROUTE
        required_cols = {"content", "post_type", "final_sentiment"}
        if not required_cols.issubset(set([c.lower() for c in df_loaded.columns])):
            st.error(
                f"❌ Dataset tidak memiliki kolom wajib sosmed: {', '.join(required_cols)}."
            )
        else:
            run_sosmed_dashboard(df_loaded.copy())
else:
    st.info("Silakan upload atau unduh ZIP untuk melihat ringkasan topik.")