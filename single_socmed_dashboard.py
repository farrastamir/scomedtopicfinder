# =========================================================
#  Topic Summary NoLimit Dashboard — ONM & Sosmed (Streamlit)
#  versi 2025-06-11  (dengan parser tanggal custom + bug-fix indentasi)
# =========================================================

import streamlit as st
import pandas as pd
import zipfile, urllib.request, re, textwrap
from collections import Counter

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Topic Summary – ONM & Sosmed",
                   page_icon="📊",
                   layout="wide")
st.title("📰 Topic Summary NoLimit Dashboard — ONM & Sosmed")

# ----------------- UTIL: URL / PLATFORM -----------------
def clean_url(u: str) -> str:
    u = str(u).strip(" \"'")
    return re.sub(r"^https:/([^/])", r"https://\1", u)

PLATFORM_COLOR = {
    "youtube": "#FF0000", "tiktok": "#000000",
    "instagram": "#C13584", "twitter": "#1DA1F2",
    "x": "#1DA1F2", "facebook": "#1877F2",
}
def platform_from_url(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)/", str(url).lower())
    dom = m.group(1) if m else ""
    for k, c in PLATFORM_COLOR.items():
        if k in dom:
            return f'<span style="color:{c};font-weight:bold">{k.capitalize()}</span>'
    return "-"

# ----------------- UTIL: PARSE TANGGAL -----------------
def parse_dates_series(s: pd.Series) -> pd.Series:
    """Parse dd/mm/yyyy hh.mm.ss atau dd/mm/yy hh.mm → datetime."""
    def norm(x: str):
        x = str(x)
        if " " in x:
            d, t = x.split(" ", 1)
            t = t.replace(".", ":")
            x = f"{d} {t}"
        return x
    cleaned = s.astype(str).map(norm)
    return pd.to_datetime(cleaned, dayfirst=True, errors="coerce")

def get_date_column(df: pd.DataFrame, prefer: list[str]) -> str | None:
    """Kembalikan nama kolom tanggal yang berhasil diparse ke datetime."""
    for col in prefer:
        if col in df.columns:
            df[col] = parse_dates_series(df[col])
            return col
    for col in df.columns:
        if re.search(r"date", col, re.I):
            df[col] = parse_dates_series(df[col])
            return col
    return None

# ----------------- UTIL: ZIP LOADER -----------------
@st.cache_data(show_spinner=False)
def extract_csv_from_zip(zfile):
    try:
        with zipfile.ZipFile(zfile) as z:
            csvs = [f for f in z.namelist() if f.endswith(".csv")]
            if not csvs:
                st.error("❌ ZIP tidak memuat CSV.")
                return pd.DataFrame()
            dfs = [pd.read_csv(z.open(f), delimiter=";", quotechar='"',
                               engine="python", on_bad_lines="skip", dtype=str)
                   for f in csvs]
            return pd.concat(dfs, ignore_index=True)
    except zipfile.BadZipFile:
        st.error("❌ ZIP rusak.")
        return pd.DataFrame()

# ----------------- UTIL: WORD FREQ + BADGE -----------------
STOPWORDS = set(pd.read_csv(
    "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt",
    header=None)[0].tolist())

def word_freq(s: pd.Series, top=500):
    toks = re.findall(r"\b\w{3,}\b", " ".join(s).lower())
    toks = [t for t in toks if t not in STOPWORDS]
    return Counter(toks).most_common(top)

def badge(s):
    clr = {"positive": "green", "negative": "red", "neutral": "gray"}.get(str(s).lower(), "black")
    return f'<span style="color:{clr};font-weight:bold">{s}</span>'

# ----------------- UTIL: ADVANCED KEYWORD -----------------
def parse_advanced_keywords(q):
    q = q.strip()
    if not q:
        return [], [], []
    inc, phr, exc = [], [], []
    for tok in re.findall(r'\"[^\"]+\"|\([^)]+\)|\S+', q):
        if tok.startswith('"'):
            phr.append(tok[1:-1])
        elif tok.startswith('-'):
            exc.extend(tok[1:].strip('()').split())
        elif tok.startswith('('):
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

# ----------------- DASHBOARD: ONM -----------------
def run_onm(df):
    df = df.rename(str.lower, axis=1)
    date_col = get_date_column(df, ["date_published"])
    for c in ["title", "body", "url", "sentiment"]:
        if c in df:
            df[c] = df[c].astype(str).str.strip("'")
    df["tier"] = df.get("tier", "-").fillna("-")
    df["label"] = df.get("label", "")

    st.sidebar.header("🎛️ Filter — ONM")
    sent_sel = st.sidebar.selectbox("Sentimen",
                                    ["All"] + sorted(df["sentiment"].str.lower().unique()))
    lab_sel = st.sidebar.selectbox("Label",
                                   ["All"] + sorted({l.strip() for sub in df["label"]
                                                     for l in str(sub).split(',') if l.strip()}))

    # --- tanggal ---
    mode = st.sidebar.radio("Filter tanggal", ["Semua", "Custom"],
                            horizontal=True, key="onm_date_mode")
    use_date = False
    d_start = d_end = None
    if mode == "Custom":
        if date_col and not df[date_col].dropna().empty:
            min_d, max_d = df[date_col].min().date(), df[date_col].max().date()
            d_start, d_end = st.sidebar.date_input("Rentang",
                                                   (min_d, max_d),
                                                   min_value=min_d,
                                                   max_value=max_d,
                                                   key="onm_date_input")
            use_date = True
        else:
            st.sidebar.info("Kolom tanggal tidak tersedia / semua NaT.")

    if st.sidebar.button("🔄 Clear", key="onm_clear"):
        st.session_state.update(onm_kw="", onm_hl="")
    kw = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="onm_kw")
    hl = st.sidebar.text_input("Highlight Kata", key="onm_hl")

    # --- filtering ---
    m = pd.Series(True, df.index)
    if sent_sel != "All":
        m &= df["sentiment"].str.lower() == sent_sel
    if lab_sel != "All":
        m &= df["label"].apply(lambda x: lab_sel in [s.strip() for s in str(x).split(',')])
    if use_date and date_col:
        m &= df[date_col].dt.date.between(d_start, d_end)
    inc, phr, exc = parse_advanced_keywords(kw)
    if kw:
        m &= (df["title"].apply(lambda x: match_advanced(x, inc, phr, exc)) |
              df["body"].apply(lambda x: match_advanced(x, inc, phr, exc)))
    filt = df[m].copy()

    # --- summary ---
    rank = {"tier 1": 0, "tier 2": 1, "tier 3": 2, "-": 3, "": 4}

    def best_link(g):
        g = g.copy()
        g["rk"] = g["tier"].str.lower().map(rank).fillna(4)
        g = g.sort_values("rk")
        return g["url"].iloc[0] if not g["url"].empty else "-"

    gpd = (filt.groupby("title", sort=False)
           .apply(best_link).to_frame("Link")
           .join(filt.groupby("title").agg(
               Total=("title", "size"),
               Sentiment=("sentiment",
                          lambda x: x.mode().iloc[0] if not x.mode().empty else "-"),
           ))
           .reset_index().sort_values("Total", ascending=False))

    gpd["Sentiment"] = gpd["Sentiment"].apply(badge)
    gpd["Link"] = gpd["Link"].apply(
        lambda u: f'<a href="{clean_url(u)}" target="_blank">Link</a>' if u != "-" else "-")

    if hl:
        pattern = re.compile("(?i)(" + "|".join(map(re.escape,
                         [h.strip('"') for h in re.findall(r'"[^"]+"|\S+', hl)])) + ")")
        gpd["title"] = gpd["title"].apply(lambda t: pattern.sub(r"<mark>\1</mark>", t))

    st.markdown("### 📊 Ringkasan Topik (ONM)")
    style = (gpd[["title", "Total", "Sentiment", "Link"]]
             .style.set_properties(subset=["Total", "Link"],
                                   **{"text-align": "center"})
             .hide(axis="index"))
    st.write(style.to_html(escape=False), unsafe_allow_html=True)

# ----------------- DASHBOARD: SOSMED -----------------
def run_sosmed(df):
    df = df.rename(str.lower, axis=1)
    df["content"] = df["content"].astype(str).str.lstrip("'")
    df["final_sentiment"] = df["final_sentiment"].astype(str).str.strip("\"' ")
    df["link"] = df["link"].apply(clean_url)
    date_col = get_date_column(df, ["date_created"])

    st.sidebar.header("🎛️ Filter — Sosmed")
    scope = st.sidebar.radio("Conversation", ("All", "Tanpa Comment", "Tanpa Post"), index=0)
    plats = sorted({p for p in df["specific_resource"].dropna() if str(p).strip()})
    plat_sel = st.sidebar.multiselect("Platform", plats, default=plats) if plats else []
    sent_sel = st.sidebar.selectbox("Sentimen",
                                    ["All"] + sorted(df["final_sentiment"].dropna().str.lower().unique()))
    mode = st.sidebar.radio("Filter tanggal", ["Semua", "Custom"],
                            horizontal=True, key="soc_date_mode")
    use_date = False
    d_start = d_end = None
    if mode == "Custom":
        if date_col and not df[date_col].dropna().empty:
            min_d, max_d = df[date_col].min().date(), df[date_col].max().date()
            d_start, d_end = st.sidebar.date_input("Rentang",
                                                   (min_d, max_d),
                                                   min_value=min_d,
                                                   max_value=max_d,
                                                   key="soc_date_input")
            use_date = True
        else:
            st.sidebar.info("Kolom tanggal tidak tersedia / semua NaT.")

    if st.sidebar.button("🔄 Clear", key="soc_clear"):
        st.session_state.update(soc_kw="", soc_hl="")
    kw = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="soc_kw")
    hl = st.sidebar.text_input("Highlight Kata", key="soc_hl")
    show_wc = st.sidebar.checkbox("WordCloud", True, key="soc_wc")

    # filtering
    m = pd.Series(True, df.index)
    if plat_sel:
        m &= df["specific_resource"].isin(plat_sel)
    if sent_sel != "All":
        m &= df["final_sentiment"].str.lower() == sent_sel
    if use_date and date_col:
        m &= df[date_col].dt.date.between(d_start, d_end)
    inc, phr, exc = parse_advanced_keywords(kw)
    if kw:
        m &= df["content"].apply(lambda x: match_advanced(x, inc, phr, exc))
    filt = df[m].copy()

    # scope split
    is_com = filt["reply_to_original_id"].notna()
    is_post = filt["post_type"].str.lower() == "post_made"
    if scope == "Tanpa Comment":
        filt_scope = filt[~is_com]
    elif scope == "Tanpa Post":
        filt_scope = filt[~is_post]
    else:
        filt_scope = filt

    comments = filt_scope[is_com]
    com_cnt = comments.groupby("reply_to_original_id").size()
    com_sent = comments.groupby("reply_to_original_id")["final_sentiment"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    posts = filt_scope[is_post].copy()
    posts["value"] = posts["original_id"].map(com_cnt).fillna(0).astype(int)
    posts["sent_final"] = posts["original_id"].map(com_sent).fillna(posts["final_sentiment"])
    talk = filt_scope[~is_post].copy()
    talk["value"] = 1
    talk["sent_final"] = talk["final_sentiment"]

    if scope == "Tanpa Post":
        base = talk
    elif scope == "Tanpa Comment":
        base = pd.concat([talk, posts], ignore_index=True)
    else:
        cm_all = comments.copy()
        cm_all["value"] = 1
        cm_all["sent_final"] = cm_all["final_sentiment"]
        base = pd.concat([talk, posts, cm_all], ignore_index=True)

    def best_link(series, val_series):
        d = pd.DataFrame({"link": series, "val": val_series}).dropna()
        d = d[d["link"].astype(str).str.strip() != ""]
        return clean_url(d.sort_values("val", ascending=False)["link"].iloc[0]) if not d.empty else "-"

    summary = (base.groupby("content", sort=False)
               .agg(Total=("value", "sum"),
                    Sentiment=("sent_final",
                               lambda x: x.mode().iloc[0] if not x.mode().empty else "-"),
                    Link=("link", lambda x: best_link(x, base.loc[x.index, "value"])))
               .sort_values("Total", ascending=False)
               .reset_index())

    summary["Short"] = summary["content"].apply(
        lambda t: textwrap.shorten(re.sub(r'<.*?>', '', t), 120, "…"))
    summary["Text"] = summary.apply(
        lambda r: f'<span title="{r.content}">{r.Short}</span>', axis=1)
    summary["Sentiment"] = summary["Sentiment"].apply(badge)
    summary["Link"] = summary["Link"].apply(
        lambda u: f'<a href="{u}" target="_blank">Link</a>' if u != "-" else "-")
    summary["Platform"] = summary["Link"].apply(
        lambda html: platform_from_url(re.search(r'href="([^"]+)"', html).group(1))
        if "href" in html else "-")

    summary = summary[["Text", "Total", "Sentiment", "Platform", "Link"]]
    if hl:
        pattern = re.compile("(?i)(" + "|".join(map(re.escape,
                         [h.strip('"') for h in re.findall(r'"[^"]+"|\S+', hl)])) + ")")
        summary["Text"] = summary["Text"].apply(lambda t: pattern.sub(r"<mark>\1</mark>", t))

    st.markdown("### 📊 Ringkasan Topik (Sosmed)")
    st.caption(f"Dataset: {len(filt):,} | Summary: {len(summary):,}")
    style = (summary.style
             .set_properties(subset=["Total", "Link", "Platform"],
                             **{"text-align": "center"})
             .hide(axis="index"))
    st.write(style.to_html(escape=False), unsafe_allow_html=True)

    if show_wc:
        wc_df = pd.DataFrame(word_freq(base["content"], 500), columns=["Kata", "Jumlah"])
        st.markdown("### ☁️ Word Cloud (Top 500)")
        st.dataframe(wc_df, use_container_width=True)

# ----------------- ENTRY -----------------
st.markdown("### 📁 Pilih sumber data")
mode = st.radio("Input ZIP via:", ["Upload File", "Link Download"], horizontal=True)
zip_data = None
if mode == "Upload File":
    up = st.file_uploader("Unggah ZIP", type="zip")
    if up:
        zip_data = up
else:
    url = st.text_input("URL ZIP")
    if st.button("Proceed") and url:
        tmp = "/tmp/data.zip"
        try:
            urllib.request.urlretrieve(url, tmp)
            zip_data = tmp
        except Exception as e:
            st.error(f"❌ Gagal unduh: {e}")

if "df" not in st.session_state:
    st.session_state.df = None
if zip_data:
    with st.spinner("📖 Membaca data…"):
        df_all = extract_csv_from_zip(zip_data)
        if not df_all.empty:
            st.session_state.df = df_all.copy()

if st.session_state.df is not None:
    df = st.session_state.df
    cols = {c.lower() for c in df.columns}
    if "tier" in cols:
        run_onm(df.copy())
    elif {"content", "post_type", "final_sentiment"}.issubset(cols):
        run_sosmed(df.copy())
    else:
        st.error("❌ Struktur kolom tidak cocok ONM atau Sosmed.")
else:
    st.info("Unggah atau tautkan ZIP terlebih dahulu.")
