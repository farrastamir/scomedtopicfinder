# =========================================================
#  Topic Summary NoLimit Dashboard — ONM & Sosmed (Streamlit)
#  2025-06-13  —  + Prompt-Template Copy Button
# =========================================================

import streamlit as st, pandas as pd, re, textwrap, zipfile, urllib.request
import streamlit.components.v1 as components   # ⬅️ butuh untuk tombol-clipboard
from collections import Counter, defaultdict

# ---------- CONFIG ----------
st.set_page_config(page_title="Topic Summary – ONM & Sosmed",
                   page_icon="📊", layout="wide")
st.title("📰 Topic Summary NoLimit Dashboard — ONM & Sosmed")

# ---------- UTIL : URL & PLATFORM ----------
def clean_url(u: str) -> str:
    u = str(u).strip(" \"'")
    return re.sub(r"^https:/([^/])", r"https://\1", u)

PLATFORM_COLOR = {"youtube": "#FF0000", "tiktok": "#000000",
                  "instagram": "#C13584", "twitter": "#1DA1F2",
                  "x": "#1DA1F2", "facebook": "#1877F2"}

def platform_color_badge(name: str) -> str:
    key = name.lower()
    for k, c in PLATFORM_COLOR.items():
        if k in key:
            return f'<span style="color:{c};font-weight:bold">{k.capitalize()}</span>'
    return f'<span style="font-weight:bold">{name}</span>'

def platform_from_url(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)/", str(url).lower())
    dom = m.group(1) if m else ""
    return platform_color_badge(dom.split(".")[0]) if dom else "-"

# ---------- UTIL : Trim & Date Parse ----------
def trim_columns(df, need: set[str]):
    for col in need:
        if col not in df.columns:
            df[col] = ""
    return df

def parse_dates_series(s):
    def norm(x):
        x = str(x)
        if " " in x:
            d, t = x.split(" ", 1)
            t = t.replace(".", ":")
            x = f"{d} {t}"
        return x
    return pd.to_datetime(s.astype(str).map(norm), dayfirst=True, errors="coerce")

def get_date_column(df, prefer: list[str]):
    for col in prefer:
        if col in df.columns:
            df[col] = parse_dates_series(df[col])
            return col
    for col in df.columns:
        if re.search(r"date", col, re.I):
            df[col] = parse_dates_series(df[col])
            return col
    return None

# ---------- UTIL : ZIP ----------
@st.cache_data
def extract_csv_from_zip(zfile):
    try:
        with zipfile.ZipFile(zfile) as z:
            files = [f for f in z.namelist() if f.endswith(".csv")]
            if not files:
                st.error("❌ ZIP tidak memuat CSV.")
                return pd.DataFrame()
            df = pd.concat([pd.read_csv(z.open(f), delimiter=";", quotechar='"',
                                        engine="python", on_bad_lines="skip", dtype=str)
                            for f in files], ignore_index=True)
            return df
    except zipfile.BadZipFile:
        st.error("❌ ZIP rusak.")
        return pd.DataFrame()

# ---------- UTIL : Word freq & Badge ----------
STOPWORDS = set(pd.read_csv(
    "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt",
    header=None)[0].tolist())

def word_freq(s, top=500):
    toks = re.findall(r"\b\w{3,}\b", " ".join(s).lower())
    toks = [t for t in toks if t not in STOPWORDS]
    return Counter(toks).most_common(top)

def badge(val):
    clr = {"positive": "green", "negative": "red", "neutral": "gray"}.get(str(val).lower(), "black")
    return f'<span style="color:{clr};font-weight:bold">{val}</span>'

# ---------- ADVANCED KEYWORD PARSER ----------
def parse_adv(q):
    q = q.strip()
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

def match(text, inc, phr, exc):
    low = str(text).lower()
    if any(w in low for w in exc):
        return False
    if any(p not in low for p in phr):
        return False
    return all(any(w in low for w in g) for g in inc)

# ---------- SAFE SHORTEN ----------
def safe_shorten(txt, width=120):
    try:
        plain = re.sub(r'<.*?>', '', str(txt))
        return textwrap.shorten(plain, width=width, placeholder="…")
    except Exception:
        return str(txt)[:width] + "…"

# ---------- PROMPT-TEMPLATE UTIL ---------------------------------------------
def generate_template(client: str, rows: list[str], mode: str) -> str:
    client = client or "[Client]"
    if mode == "sosmed":
        header = (f"Anda adalah seorang data analyst yang bergerak di bidang Sosial Media. "
                  f"Anda memiliki tugas untuk mencari topik pembicaraan mengenai {client}. "
                  "Dibawah ini terdapat table yang saya copy untuk anda yang berisikan perckapan sosial media "
                  f"mengenai {client} tersebut dan juga jumlah percakapan pada spesifik pembicaraan tersebut. "
                  "Tugas anda adalah merangkum … (instruksi selengkapnya sama seperti arahan Anda).")
    else:
        header = (f"Anda adalah seorang data analyst yang bergerak di bidang Media Daring. "
                  f"Anda memiliki tugas untuk mencari topik pemberitaan mengenai {client}. "
                  "Dibawah ini terdapat table yang saya copy untuk anda … (instruksi selengkapnya).")
    rows_txt = "\n".join(f"- {r}" for r in rows)
    return f"{header}\n\n{rows_txt}"

def copy_template_component(text: str, uid: str):
    components.html(f"""
    <div style="margin-bottom:6px">
      <textarea id="tmpl_{uid}" style="width:100%;height:300px">{text}</textarea><br>
      <button style="margin-top:4px;padding:6px 12px;cursor:pointer"
              onclick="navigator.clipboard.writeText(
                       document.getElementById('tmpl_{uid}').value);
                       alert('📋 Template tersalin!');">
              📋 Copy Template
      </button>
    </div>
    """, height=330)

# ---------- DASHBOARD : ONM ---------------------------------------------------
def run_onm(df):
    need = {"title", "body", "url", "tier", "sentiment", "label", "date_published"}
    df = trim_columns(df.rename(str.lower, axis=1), need)
    date_col = get_date_column(df, ["date_published"])
    for c in ["title", "body", "url", "sentiment"]:
        df[c] = df[c].astype(str).str.strip("'")
    df["tier"] = df["tier"].fillna("-")
    df["label"] = df["label"].fillna("")

    # ---------- SIDEBAR FILTER ----------
    st.sidebar.header("🎛️ Filter — ONM")
    stats_box = st.sidebar.empty()

    sent_sel = st.sidebar.selectbox("Sentimen", ["All"] + sorted(df["sentiment"].str.lower().unique()))
    lab_sel  = st.sidebar.selectbox("Label", ["All"] + sorted({l.strip() for sub in df["label"]
                                                              for l in str(sub).split(',') if l.strip()}))
    mode = st.sidebar.radio("Filter tanggal", ["Semua", "Custom"], horizontal=True, key="onm_date_mode")
    use_date = False
    if mode == "Custom" and date_col and not df[date_col].dropna().empty:
        d_start, d_end = st.sidebar.date_input(
            "Rentang", (df[date_col].min().date(), df[date_col].max().date()),
            key="onm_date_input")
        use_date = True

    if st.sidebar.button("🔄 Clear", key="onm_clear"):
        st.session_state.update(onm_kw="", onm_hl="")

    kw = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="onm_kw")
    hl = st.sidebar.text_input("Highlight Kata", key="onm_hl")

    # ---------- APPLY FILTER ----------
    m = pd.Series(True, df.index)
    if sent_sel != "All": m &= df["sentiment"].str.lower() == sent_sel
    if lab_sel  != "All": m &= df["label"].apply(lambda x: lab_sel in [s.strip() for s in str(x).split(',')])
    if use_date and date_col: m &= df[date_col].dt.date.between(d_start, d_end)
    inc, phr, exc = parse_adv(kw)
    if kw:
        m &= (df["title"].apply(lambda x: match(x, inc, phr, exc)) |
              df["body"].apply(lambda x: match(x, inc, phr, exc)))
    filt = df[m]

    # ---------- TEMPLATE GENERATOR (selalu muncul sebelum tabel) ----------
    st.markdown("#### 🔖 Copy Prompt Template (ONM)")
    client_name = st.text_input("Nama Client", key="client_onm")
    tmpl_text   = generate_template(client_name, filt["title"].head(100).tolist(), "onm")
    copy_template_component(tmpl_text, "onm")

    # ---------- SIDEBAR STATS ----------
    sl = filt["sentiment"].str.lower()
    stats_box.markdown(f"""
    <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
      <span style='font-size:24px'>📊</span>
      <span style='font-size:26px;font-weight:bold'>{len(filt):,} Article</span>
    </div>
    <div style='font-size:15px;'>
      <span style='color:green;font-weight:bold;'>🟢 {(sl=='positive').sum()}</span> |
      <span style='color:gray;font-weight:bold;'>⚪ {(sl=='neutral').sum()}</span> |
      <span style='color:red;font-weight:bold;'>🔴 {(sl=='negative').sum()}</span>
    </div>
    """, unsafe_allow_html=True)

    # ---------- RINGKASAN TOPIK ----------
    rank = {"tier 1":0,"tier 2":1,"tier 3":2,"-":3,"":4}
    def best_link(g):
        g = g.copy()
        g["rk"] = g["tier"].str.lower().map(rank).fillna(4)
        return g.sort_values("rk")["url"].iloc[0] if not g.empty else "-"

    gpd = (filt.groupby("title", sort=False)
           .apply(best_link).to_frame("Link")
           .join(filt.groupby("title")
                 .agg(Total=("title","size"),
                      Sentiment=("sentiment", lambda x: x.mode().iloc[0] if not x.mode().empty else "-")))
           .reset_index().sort_values("Total", ascending=False))

    gpd["Sentiment"] = gpd["Sentiment"].apply(badge)
    gpd["Link"]      = gpd["Link"].apply(lambda u: f'<a href="{clean_url(u)}" target="_blank">Link</a>' if u!="-"
                                         else "-")
    if hl:
        pat = re.compile("(?i)("+"|".join(map(re.escape,
                       [h.strip('"') for h in re.findall(r'"[^"]+"|\S+', hl)]))+")")
        gpd["title"] = gpd["title"].apply(lambda t: pat.sub(r"<mark>\\1</mark>", t))

    st.markdown("### 📊 Ringkasan Topik (ONM)")
    st.write(gpd[["title","Total","Sentiment","Link"]]
             .style.set_properties(subset=["Total","Link"], **{"text-align":"center"})
             .hide(axis="index").to_html(escape=False), unsafe_allow_html=True)

# ---------- DASHBOARD : SOSMED -----------------------------------------------
def run_sosmed(df):
    need = {"content","post_type","final_sentiment","specific_resource",
            "specific_resource_type","reply_to_original_id","original_id",
            "link","date_created"}
    df = trim_columns(df.rename(str.lower, axis=1), need)
    df["content"] = df["content"].astype(str).str.lstrip("'")
    df["final_sentiment"] = df["final_sentiment"].astype(str).str.strip("\"' ")
    df["link"] = df["link"].apply(clean_url)
    date_col = get_date_column(df, ["date_created"])

    plat_col = ("specific_resource_type"
                if df["specific_resource_type"].str.strip().any()
                else "specific_resource")

    # ---------- SIDEBAR FILTER ----------
    st.sidebar.header("🎛️ Filter — Sosmed")
    stats_box = st.sidebar.empty()

    scope = st.sidebar.radio("Conversation", ("All","Tanpa Comment","Tanpa Post"), key="soc_scope")

    plat_values = sorted({p for p in df[plat_col].dropna() if str(p).strip()})
    sel_plat=[]
    if plat_values:
        st.sidebar.markdown("**Platform**")
        cols = st.sidebar.columns(len(plat_values))
        for i,p in enumerate(plat_values):
            if cols[i].checkbox(p,True,key=f"plat_{p}",label_visibility="collapsed"): sel_plat.append(p)
            cols[i].markdown(platform_color_badge(p), unsafe_allow_html=True)

    sent_sel = st.sidebar.selectbox("Sentimen", ["All"] + sorted(df["final_sentiment"].str.lower().unique()))
    mode = st.sidebar.radio("Filter tanggal", ["Semua","Custom"], horizontal=True, key="soc_date_mode")
    use_date=False
    if mode=="Custom" and date_col and not df[date_col].dropna().empty:
        d_start,d_end = st.sidebar.date_input(
            "Rentang", (df[date_col].min().date(), df[date_col].max().date()),
            key="soc_date_input"); use_date=True

    if st.sidebar.button("🔄 Clear", key="soc_clear"):
        st.session_state.update(soc_kw="", soc_hl="")

    kw = st.sidebar.text_input("Kata kunci (\"frasa\" -exclude)", key="soc_kw")
    hl = st.sidebar.text_input("Highlight Kata", key="soc_hl")
    show_wc = st.sidebar.checkbox("WordCloud", True, key="soc_wc")

    # ---------- APPLY FILTER ----------
    m = pd.Series(True, df.index)
    if sel_plat: m &= df[plat_col].isin(sel_plat)
    if sent_sel!="All": m &= df["final_sentiment"].str.lower()==sent_sel
    if use_date and date_col: m &= df[date_col].dt.date.between(d_start,d_end)
    inc,phr,exc = parse_adv(kw)
    if kw: m &= df["content"].apply(lambda x: match(x,inc,phr,exc))
    filt = df[m]

    # ---------- TEMPLATE GENERATOR ----------
    st.markdown("#### 🔖 Copy Prompt Template (Sosmed)")
    client_name = st.text_input("Nama Client", key="client_soc")
    tmpl_text   = generate_template(client_name, filt["content"].head(100).tolist(), "sosmed")
    copy_template_component(tmpl_text, "soc")

    # ---------- SIDEBAR STATS ----------
    sl = filt["final_sentiment"].str.lower()
    stats_box.markdown(f"""
    <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
      <span style='font-size:24px'>📊</span>
      <span style='font-size:26px;font-weight:bold'>{len(filt):,} Percakapan</span>
    </div>
    <div style='font-size:15px;'>
      <span style='color:green;font-weight:bold;'>🟢 {(sl=='positive').sum()}</span> |
      <span style='color:gray;font-weight:bold;'>⚪ {(sl=='neutral').sum()}</span> |
      <span style='color:red;font-weight:bold;'>🔴 {(sl=='negative').sum()}</span>
    </div>
    """, unsafe_allow_html=True)

    # ---------- CONVERSATION SCOPE BUILD ----------
    is_com  = filt["reply_to_original_id"].notna()
    is_post = filt["post_type"].str.lower()=="post_made"

    if scope=="Tanpa Comment": filt_scope=filt[~is_com]
    elif scope=="Tanpa Post": filt_scope=filt[~is_post]
    else: filt_scope=filt

    comments = filt_scope[is_com]
    com_cnt  = comments.groupby("reply_to_original_id").size()
    com_sent = comments.groupby("reply_to_original_id")["final_sentiment"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    posts = filt_scope[is_post].copy()
    posts["value"] = posts["original_id"].map(com_cnt).fillna(0).astype(int)
    posts["sent_final"] = posts["original_id"].map(com_sent).fillna(posts["final_sentiment"])

    talk = filt_scope[~is_post].copy()
    talk["value"] = 1
    talk["sent_final"]=talk["final_sentiment"]

    if scope=="Tanpa Post": base=talk
    elif scope=="Tanpa Comment": base=pd.concat([talk,posts],ignore_index=True)
    else:
        c_all=comments.copy(); c_all["value"]=1; c_all["sent_final"]=c_all["final_sentiment"]
        base=pd.concat([talk,posts,c_all],ignore_index=True)

    # ---------- SUMMARY ----------
    def best_link(series,val):
        d=pd.DataFrame({"link":series,"val":val}).dropna()
        d=d[d["link"].astype(str).str.strip()!=""]
        return clean_url(d.sort_values("val",ascending=False)["link"].iloc[0]) if not d.empty else "-"

    summary=(base.groupby("content", sort=False)
             .agg(Total=("value","sum"),
                  Sentiment=("sent_final",lambda x: x.mode().iloc[0] if not x.mode().empty else "-"),
                  Link=("link", lambda x: best_link(x, base.loc[x.index,"value"])))
             .sort_values("Total",ascending=False).reset_index())

    summary["Short"] = summary["content"].apply(safe_shorten)
    summary["Text"]  = summary.apply(lambda r: f'<span title="{str(r.content)}">{r.Short}</span>', axis=1)
    summary["Sentiment"]=summary["Sentiment"].apply(badge)
    summary["Link"]=summary["Link"].apply(lambda u: f'<a href="{u}" target="_blank">Link</a>' if u!="-"
                                          else "-")
    summary["Platform"]=summary["Link"].apply(
        lambda html: platform_from_url(re.search(r'href="([^"]+)"', html).group(1))
        if "href" in html else "-")

    summary = summary[["Text","Total","Sentiment","Platform","Link"]]
    if hl:
        pat=re.compile("(?i)("+"|".join(map(re.escape,
                     [h.strip('"') for h in re.findall(r'"[^"]+"|\S+', hl)]))+")")
        summary["Text"]=summary["Text"].apply(lambda t: pat.sub(r"<mark>\\1</mark>", t))

    st.markdown("### 📊 Ringkasan Topik (Sosmed)")
    st.caption(f"Dataset: {len(filt):,} | Summary: {len(summary):,}")
    st.write(summary.style.set_properties(subset=["Total","Link","Platform"],
                                          **{"text-align":"center"})
             .hide(axis="index").to_html(escape=False), unsafe_allow_html=True)

    # ---------- WORD CLOUD ----------
    if show_wc:
        wc_df=pd.DataFrame(word_freq(base["content"],500),columns=["Kata","Jumlah"])
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ☁️ Word Cloud (Top 500)")
        st.sidebar.dataframe(wc_df, use_container_width=True)

# ---------- ENTRY ------------------------------------------------------------
st.markdown("### 📁 Pilih sumber data")
mode = st.radio("Input ZIP via:", ["Upload File", "Link Download"], horizontal=True)
zip_data = None
if mode=="Upload File":
    up=st.file_uploader("Unggah ZIP", type="zip")
    if up: zip_data=up
else:
    url=st.text_input("URL ZIP")
    if st.button("Proceed") and url:
        tmp="/tmp/data.zip"
        try: urllib.request.urlretrieve(url,tmp); zip_data=tmp
        except Exception as e: st.error(f"❌ Gagal unduh: {e}")

if "df" not in st.session_state: st.session_state.df=None
if zip_data:
    with st.spinner("📖 Membaca data …"):
        df_all=extract_csv_from_zip(zip_data)
        if not df_all.empty: st.session_state.df=df_all.copy()

if st.session_state.df is not None:
    df=st.session_state.df
    cols={c.lower() for c in df.columns}
    if "tier" in cols:
        run_onm(df.copy())
    elif {"content","post_type","final_sentiment"}.issubset(cols):
        run_sosmed(df.copy())
    else:
        st.error("❌ Struktur kolom tidak cocok (ONM/Sosmed).")
else:
    st.info("Unggah atau tautkan ZIP terlebih dahulu.")
