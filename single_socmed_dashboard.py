# =========================================================
#  Topic Summary NoLimit Dashboard — ONM & Sosmed (Streamlit)
#  2025-06-17  —  Integrated with Gemini & News Popup
# =========================================================

import streamlit as st, pandas as pd, re, textwrap, zipfile, urllib.request
import streamlit.components.v1 as components
import google.generativeai as genai
from collections import Counter

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
    # Fallback if preferred columns are not found
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
@st.cache_data
def load_stopwords():
    try:
        return set(pd.read_csv(
            "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt",
            header=None)[0].tolist())
    except Exception:
        return set() # Return empty set if download fails
STOPWORDS = load_stopwords()

def word_freq(s, top=500):
    toks = re.findall(r"\b\w{3,}\b", " ".join(s).lower())
    toks = [t for t in toks if t not in STOPWORDS]
    return Counter(toks).most_common(top)

def badge(val):
    clr = {"positive": "green", "negative": "red", "neutral": "gray"}.get(str(val).lower(), "black")
    return f'<span style="color:{clr};font-weight:bold">{val}</span>'

# ---------- ADVANCED KEYWORD PARSER V2 ----------
def parse_adv(q: str):
    q = q.strip()
    tokens = re.findall(r'-\([^)]+\)|-"[^"]+"|\([^)]+\)|"[^"]+"|\S+', q)
    includes, phrases, exclude_phrases, exclude_groups = [], [], [], []
    for tok in tokens:
        if tok.startswith('-"') and tok.endswith('"'):
            exclude_phrases.append(tok[2:-1].lower())
        elif tok.startswith('-(') and tok.endswith(')'):
            inner_q = tok[2:-1]
            ex_items = [item.strip().lower().replace('"', '') for item in re.findall(r'"[^"]+"|\w+', inner_q)]
            if ex_items: exclude_groups.append(ex_items)
        elif tok.startswith('-'):
            exclude_groups.append([tok[1:].lower()])
        elif tok.startswith('"') and tok.endswith('"'):
            phrases.append(tok[1:-1].lower())
        elif tok.startswith('(') and tok.endswith(')'):
            inc_items = [w.strip().lower() for w in tok[1:-1].split('OR') if w.strip()]
            if inc_items: includes.append(inc_items)
        else:
            includes.append([tok.lower()])
    return includes, phrases, exclude_phrases, exclude_groups

def match(text: str, includes: list, phrases: list, exclude_phrases: list, exclude_groups: list) -> bool:
    low = str(text).lower()
    if any(p in low for p in exclude_phrases): return False
    for group in exclude_groups:
        if any(w in low for w in group): return False
    if not all(p in low for p in phrases): return False
    if not all(any(w in low for w in g) for g in includes): return False
    return True

# ---------- SAFE SHORTEN ----------
def safe_shorten(txt, width=120):
    try:
        plain = re.sub(r'<.*?>', '', str(txt))
        return textwrap.shorten(plain, width=width, placeholder="…")
    except Exception:
        return str(txt)[:width] + "…"

# ---------- DASHBOARD : ONM ---------------------------------------------------
def run_onm(df, model):
    need = {"title", "body", "url", "tier", "sentiment", "label", "date_published", "source_name", "pr_value"}
    df = trim_columns(df, need)
    date_col = get_date_column(df, ["date_published"])
    for c in ["title", "body", "url", "sentiment", "source_name"]: df[c] = df[c].astype(str).str.strip("'")
    df["tier"] = df["tier"].fillna("-")
    df["label"] = df["label"].fillna("")
    df["pr_value"] = pd.to_numeric(df["pr_value"], errors='coerce').fillna(0)

    if 'selected_onm_title' not in st.session_state:
        st.session_state.selected_onm_title = None

    st.sidebar.header("🎛️ Filter — ONM")
    stats_box = st.sidebar.empty()
    sent_sel = st.sidebar.selectbox("Sentimen", ["All"] + sorted(df["sentiment"].str.lower().unique()))
    lab_sel = st.sidebar.selectbox("Label", ["All"] + sorted({l.strip() for sub in df["label"] for l in str(sub).split(',') if l.strip()}))
    mode = st.sidebar.radio("Filter tanggal", ["Semua", "Custom"], horizontal=True, key="onm_date_mode")
    use_date = False
    if mode == "Custom" and date_col and not df[date_col].dropna().empty:
        min_date, max_date = df[date_col].min(), df[date_col].max()
        d_start, d_end = st.sidebar.date_input("Rentang", (min_date, max_date), min_value=min_date, max_value=max_date, key="onm_date_input")
        use_date = True
    if st.sidebar.button("🔄 Clear", key="onm_clear"): st.session_state.update(onm_kw="", onm_hl="", onm_summary="")
    kw = st.sidebar.text_input("Kata kunci (mendukung \"frasa\" dan -exclude)", key="onm_kw")
    hl = st.sidebar.text_input("Highlight Kata", key="onm_hl")
    show_wc = st.sidebar.checkbox("WordCloud", True, key="onm_wc")

    m = pd.Series(True, df.index)
    if sent_sel != "All": m &= df["sentiment"].str.lower() == sent_sel
    if lab_sel != "All": m &= df["label"].apply(lambda x: lab_sel in [s.strip() for s in str(x).split(',')])
    if use_date and date_col: m &= df[date_col].dt.date.between(d_start, d_end)
    if kw:
        includes, phrases, exclude_phrases, exclude_groups = parse_adv(kw)
        m &= (df["title"].apply(lambda x: match(x, includes, phrases, exclude_phrases, exclude_groups)) |
              df["body"].apply(lambda x: match(x, includes, phrases, exclude_phrases, exclude_groups)))
    filt = df[m]

    if st.session_state.selected_onm_title:
        @st.dialog("Detail Berita")
        def show_news_popup(title_to_show):
            article_group = filt[filt['title'] == title_to_show].reset_index()
            if article_group.empty:
                st.warning("Artikel tidak ditemukan. Filter mungkin telah berubah.")
                if st.button("Tutup"):
                    st.rerun()
                return

            sample = article_group.iloc[0]
            title_display = sample['title']
            body_display = sample['body'] if pd.notna(sample['body']) else "Konten tidak tersedia."
            
            if hl:
                pat = re.compile("(?i)("+"|".join(map(re.escape, [h.strip('"') for h in re.findall(r'"[^"]+"|\S+', hl)]))+")")
                title_display = pat.sub(r"<mark>\1</mark>", title_display)
                body_display = pat.sub(r"<mark>\1</mark>", body_display)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"### {title_display}", unsafe_allow_html=True)
                st.markdown(f"**Media:** `{sample['source_name']}`")
                
                date_str = sample[date_col].strftime('%d %B %Y, %H:%M') if date_col and pd.notna(sample[date_col]) else "Tidak ada tanggal"
                pr_val_str = f"Rp {sample['pr_value']:,.0f}" if sample['pr_value'] > 0 else "Tidak ada PR Value"
                st.markdown(f"**Tanggal Terbit:** `{date_str}` | **PR Value:** `{pr_val_str}`")
                
                st.markdown("---")
                with st.container(height=400):
                    st.markdown(body_display, unsafe_allow_html=True)
            with col2:
                st.markdown("#### Media Lainnya")
                st.caption("Daftar media unik yang memberitakan topik yang sama.")
                media_list = article_group['source_name'].unique()
                st.dataframe(pd.DataFrame(media_list, columns=["Nama Media"]), use_container_width=True, hide_index=True)
        
        show_news_popup(st.session_state.selected_onm_title)
        st.session_state.selected_onm_title = None
        st.rerun()

    st.markdown("#### ✨ Ringkasan Otomatis (AI)")
    rank = {"tier 1":0, "tier 2":1, "tier 3":2, "-":3, "":4}
    def best_link(g):
        g = g.copy()
        g["rk"] = g["tier"].str.lower().map(rank).fillna(4)
        return g.sort_values("rk")["url"].iloc[0] if not g.empty else "-"
    
    if not filt.empty:
        gpd = (filt.groupby("title", sort=False).apply(best_link).to_frame("Link")
               .join(filt.groupby("title").agg(Total=("title","size"), Sentiment=("sentiment", lambda x: x.mode().iloc[0] if not x.mode().empty else "-"))))
        gpd = gpd.reset_index().sort_values("Total", ascending=False)
    else:
        gpd = pd.DataFrame(columns=["title", "Link", "Total", "Sentiment"])
        
    client_name = st.text_input("Nama Client", key="client_onm")
    if st.button("Buat Ringkasan Topik dengan Gemini", key="gen_onm"):
        if not model:
            st.warning("API Key Gemini belum dimasukkan di sidebar.")
        elif gpd.empty:
            st.warning("Tidak ada data untuk diringkas. Sesuaikan filter Anda.")
        else:
            top_10_topics = gpd.head(10)
            topics_list = [f"- {row['title']} ({row['Total']} artikel)" for index, row in top_10_topics.iterrows()]
            topics_text = "\n".join(topics_list)
            prompt = f"""Anda adalah seorang data analyst yang bergerak di bidang Media Daring (Online News Media). Tugas Anda adalah merangkum topik pemberitaan utama mengenai klien bernama "{client_name}". Di bawah ini adalah daftar 10 topik teratas beserta jumlah artikelnya:\n{topics_text}\n\nBerdasarkan data di atas, buatlah sebuah ringkasan eksekutif (executive summary) dalam format poin-poin atau narasi singkat yang menyoroti isu-isu utama yang paling banyak dibicarakan. Fokus pada apa yang menjadi sorotan utama media."""
            with st.spinner("Gemini sedang menganalisis dan membuat ringkasan..."):
                try:
                    response = model.generate_content(prompt)
                    st.session_state.onm_summary = response.text
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                        st.error("⚠️ Gagal: Anda telah mencapai batas kuota gratis harian untuk API Gemini. Coba lagi besok.")
                    else:
                        st.error(f"Terjadi kesalahan saat menghubungi Gemini: {e}")
    
    if 'onm_summary' in st.session_state and st.session_state.onm_summary:
        with st.expander("📝 **Lihat Ringkasan Eksekutif dari Gemini**", expanded=True):
            st.markdown(st.session_state.onm_summary)
            if st.button("Hapus Ringkasan", key="clear_onm_summary"):
                st.session_state.onm_summary = ""
                st.rerun()

    sl = filt["sentiment"].str.lower()
    stats_box.markdown(f"""<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'><span style='font-size:24px'>📊</span><span style='font-size:26px;font-weight:bold'>{len(filt):,} Artikel</span></div><div style='font-size:15px;'><span style='color:green;font-weight:bold;'>🟢 {(sl=='positive').sum()}</span> | <span style='color:gray;font-weight:bold;'>⚪ {(sl=='neutral').sum()}</span> | <span style='color:red;font-weight:bold;'>🔴 {(sl=='negative').sum()}</span></div>""", unsafe_allow_html=True)
    
    st.markdown("### 📊 Ringkasan Topik (ONM)")
    header_cols = st.columns([0.5, 6, 1, 1.5, 1])
    header_cols[0].markdown("**Detail**")
    header_cols[1].markdown("**Judul Topik**")
    header_cols[2].markdown("<div style='text-align: center;'>Total</div>", unsafe_allow_html=True)
    header_cols[3].markdown("<div style='text-align: center;'>Sentimen</div>", unsafe_allow_html=True)
    header_cols[4].markdown("<div style='text-align: center;'>Link</div>", unsafe_allow_html=True)
    st.markdown("---")

    if not gpd.empty:
        gpd["Sentiment"] = gpd["Sentiment"].apply(badge)
        gpd["Link"] = gpd["Link"].apply(lambda u: f'<a href="{clean_url(u)}" target="_blank">Teratas</a>' if u!="-" else "-")
        
        if hl:
            pat = re.compile("(?i)("+"|".join(map(re.escape, [h.strip('"') for h in re.findall(r'"[^"]+"|\S+', hl)]))+")")
            gpd["title"] = gpd["title"].apply(lambda t: pat.sub(r"<mark>\1</mark>", t))

        for index, row in gpd.iterrows():
            row_cols = st.columns([0.5, 6, 1, 1.5, 1])
            if row_cols[0].button("📰", key=f"onm_detail_{index}", help="Klik untuk melihat detail berita"):
                # Simpan judul asli (tanpa HTML) untuk pencarian
                original_title = filt[filt.index == row.name]['title'].iloc[0] if 'name' in row else gpd.loc[index, 'title']
                st.session_state.selected_onm_title = original_title
                st.rerun()
            
            row_cols[1].markdown(row['title'], unsafe_allow_html=True)
            row_cols[2].markdown(f"<div style='text-align: center;'>{row['Total']}</div>", unsafe_allow_html=True)
            row_cols[3].markdown(f"<div style='text-align: center;'>{row['Sentiment']}</div>", unsafe_allow_html=True)
            row_cols[4].markdown(f"<div style='text-align: center;'>{row['Link']}</div>", unsafe_allow_html=True)
    else:
        st.info("Tidak ada data topik untuk ditampilkan sesuai filter yang dipilih.")

    if show_wc:
        st.sidebar.markdown("---"); st.sidebar.markdown("### ☁️ Word Cloud (Top 500)")
        wc_dynamic = st.sidebar.checkbox("Wordcloud Dinamis (mengikuti filter)", value=False, key="onm_wc_dynamic")
        if wc_dynamic:
            st.sidebar.caption("Mode: Dinamis (berdasarkan data terfilter)"); source_df = filt
        else:
            st.sidebar.caption("Mode: Statis (berdasarkan data utama)"); source_df = df
        
        if not source_df.empty:
            wc_data = word_freq(source_df["title"].astype(str).tolist() + source_df["body"].astype(str).tolist(), 500)
            if wc_data:
                wc_df = pd.DataFrame(wc_data, columns=["Kata","Jumlah"])
                st.sidebar.dataframe(wc_df, use_container_width=True)
            else:
                st.sidebar.write("Tidak ada kata untuk ditampilkan.")
        else:
            st.sidebar.write("Tidak ada kata untuk ditampilkan.")

# ---------- DASHBOARD : SOSMED -----------------------------------------------
def run_sosmed(df, model):
    need = {"content","post_type","final_sentiment","specific_resource", "specific_resource_type","reply_to_original_id","original_id", "link","date_created"}
    df = trim_columns(df, need)
    df["content"] = df["content"].astype(str).str.lstrip("'")
    df["final_sentiment"] = df["final_sentiment"].astype(str).str.strip("\"' ")
    df["link"] = df["link"].apply(clean_url)
    date_col = get_date_column(df, ["date_created"])
    plat_col = "specific_resource_type" if df["specific_resource_type"].str.strip().any() else "specific_resource"

    st.sidebar.header("🎛️ Filter — Sosmed")
    stats_box = st.sidebar.empty()
    scope = st.sidebar.radio("Conversation", ("All","Tanpa Comment","Tanpa Post"), key="soc_scope")
    plat_values = sorted({str(p).strip() for p in df[plat_col].dropna() if str(p).strip()})
    sel_plat = []
    if plat_values:
        st.sidebar.markdown("**Platform**")
        num_cols = min(len(plat_values), 4) # Max 4 columns for platforms
        cols = st.sidebar.columns(num_cols)
        for i, p in enumerate(plat_values):
            col_idx = i % num_cols
            with cols[col_idx]:
                if st.checkbox(p, True, key=f"plat_{p}", label_visibility="collapsed"):
                    sel_plat.append(p)
                st.markdown(platform_color_badge(p), unsafe_allow_html=True)

    sent_sel = st.sidebar.selectbox("Sentimen", ["All"] + sorted(df["final_sentiment"].str.lower().unique()))
    mode = st.sidebar.radio("Filter tanggal", ["Semua","Custom"], horizontal=True, key="soc_date_mode")
    use_date = False
    if mode=="Custom" and date_col and not df[date_col].dropna().empty:
        min_date, max_date = df[date_col].min(), df[date_col].max()
        d_start, d_end = st.sidebar.date_input("Rentang", (min_date, max_date), min_value=min_date, max_value=max_date, key="soc_date_input")
        use_date=True
    if st.sidebar.button("🔄 Clear", key="soc_clear"): st.session_state.update(soc_kw="", soc_hl="", soc_summary="")
    kw = st.sidebar.text_input("Kata kunci (mendukung \"frasa\" dan -exclude)", key="soc_kw")
    hl = st.sidebar.text_input("Highlight Kata", key="soc_hl")
    show_wc = st.sidebar.checkbox("WordCloud", True, key="soc_wc")

    m = pd.Series(True, df.index)
    if sel_plat: m &= df[plat_col].isin(sel_plat)
    if sent_sel!="All": m &= df["final_sentiment"].str.lower()==sent_sel
    if use_date and date_col: m &= df[date_col].dt.date.between(d_start,d_end)
    if kw:
        includes, phrases, exclude_phrases, exclude_groups = parse_adv(kw)
        m &= df["content"].apply(lambda x: match(x, includes, phrases, exclude_phrases, exclude_groups))
    filt = df[m].copy()

    is_com  = filt["reply_to_original_id"].notna() & filt["reply_to_original_id"].ne('')
    is_post = filt["post_type"].str.lower()=="post_made"

    if scope=="Tanpa Comment":
        filt_scope = filt[~is_com]
    elif scope=="Tanpa Post":
        filt_scope = filt[~is_post]
    else:
        filt_scope = filt

    comments = filt_scope[is_com]
    com_cnt  = comments.groupby("reply_to_original_id").size()
    com_sent = comments.groupby("reply_to_original_id")["final_sentiment"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    posts = filt_scope[is_post].copy()
    posts["value"] = posts["original_id"].map(com_cnt).fillna(0).astype(int)
    posts["sent_final"] = posts["original_id"].map(com_sent).fillna(posts["final_sentiment"])
    
    talk = filt_scope[~is_post].copy()
    talk["value"] = 1
    talk["sent_final"]=talk["final_sentiment"]

    if scope=="Tanpa Post":
        base = talk
    elif scope=="Tanpa Comment":
        base = pd.concat([talk,posts],ignore_index=True)
    else:
        c_all=comments.copy()
        c_all["value"]=1
        c_all["sent_final"]=c_all["final_sentiment"]
        base = pd.concat([talk,posts,c_all],ignore_index=True)

    def best_link(series,val):
        d = pd.DataFrame({"link":series,"val":val}).dropna()
        d = d[d["link"].astype(str).str.strip()!=""]
        return clean_url(d.sort_values("val",ascending=False)["link"].iloc[0]) if not d.empty else "-"

    if not base.empty:
        summary=(base.groupby("content", sort=False).agg(Total=("value","sum"), Sentiment=("sent_final",lambda x: x.mode().iloc[0] if not x.mode().empty else "-"), Link=("link", lambda x: best_link(x, base.loc[x.index,"value"])) ).sort_values("Total",ascending=False).reset_index())
        summary["Short"] = summary["content"].apply(safe_shorten)
        summary["Text"]  = summary.apply(lambda r: f'<span title="{str(r.content)}">{r.Short}</span>', axis=1)
        summary["Sentiment"] = summary["Sentiment"].apply(badge)
        summary["Link"] = summary["Link"].apply(lambda u: f'<a href="{u}" target="_blank">Link</a>' if u!="-" else "-")
        summary["Platform"] = summary["Link"].apply(lambda html: platform_from_url(re.search(r'href="([^"]+)"', html).group(1)) if "href" in html else "-")
        summary = summary[["Text","Total","Sentiment","Platform","Link"]]
        if hl:
            pat = re.compile("(?i)("+"|".join(map(re.escape, [h.strip('"') for h in re.findall(r'"[^"]+"|\S+', hl)]))+")")
            summary["Text"] = summary["Text"].apply(lambda t: pat.sub(r"<mark>\1</mark>", t))
    else:
        summary = pd.DataFrame(columns=["Text","Total","Sentiment","Platform","Link"])

    st.markdown("#### ✨ Ringkasan Otomatis (AI)")
    client_name = st.text_input("Nama Client", key="client_soc")
    if st.button("Buat Ringkasan Percakapan dengan Gemini", key="gen_soc"):
        if not model:
            st.warning("API Key Gemini belum dimasukkan di sidebar.")
        elif summary.empty:
            st.warning("Tidak ada data untuk diringkas. Sesuaikan filter Anda.")
        else:
            top_10_talks = summary.head(10)
            def clean_html(raw_html): return re.sub(r'<.*?>', '', raw_html)
            talks_list = [f"- Percakapan tentang \"{clean_html(row['Text'])}\" ({row['Total']} percakapan)" for index, row in top_10_talks.iterrows()]
            talks_text = "\n".join(talks_list)
            prompt = f"""Anda adalah seorang data analyst yang bergerak di bidang Sosial Media. Tugas Anda adalah mencari topik percakapan utama mengenai klien bernama "{client_name}". Di bawah ini adalah daftar 10 percakapan teratas beserta jumlahnya:\n{talks_text}\n\nBerdasarkan data di atas, buatlah sebuah rangkuman singkat (summary) dalam format poin-poin yang menyoroti topik-topik apa saja yang sedang ramai dibicarakan oleh netizen. Gunakan bahasa yang lebih santai dan fokus pada sentimen atau isu utama dalam percakapan."""
            with st.spinner("Gemini sedang menyimak percakapan dan membuat ringkasan..."):
                try:
                    response = model.generate_content(prompt)
                    st.session_state.soc_summary = response.text
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                        st.error("⚠️ Gagal: Anda telah mencapai batas kuota gratis harian untuk API Gemini. Coba lagi besok.")
                    else:
                        st.error(f"Terjadi kesalahan saat menghubungi Gemini: {e}")

    if 'soc_summary' in st.session_state and st.session_state.soc_summary:
        with st.expander("📝 **Lihat Ringkasan Percakapan dari Gemini**", expanded=True):
            st.markdown(st.session_state.soc_summary)
            if st.button("Hapus Ringkasan", key="clear_soc_summary"):
                st.session_state.soc_summary = ""
                st.rerun()

    sl = filt["final_sentiment"].str.lower()
    stats_box.markdown(f"""<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'><span style='font-size:24px'>📊</span><span style='font-size:26px;font-weight:bold'>{len(filt):,} Percakapan</span></div><div style='font-size:15px;'><span style='color:green;font-weight:bold;'>🟢 {(sl=='positive').sum()}</span> | <span style='color:gray;font-weight:bold;'>⚪ {(sl=='neutral').sum()}</span> | <span style='color:red;font-weight:bold;'>🔴 {(sl=='negative').sum()}</span></div>""", unsafe_allow_html=True)
    st.markdown("### 📊 Ringkasan Topik (Sosmed)")
    st.caption(f"Dataset: {len(filt):,} | Summary: {len(summary):,}")
    st.write(summary.style.set_properties(subset=["Total","Link","Platform"], **{"text-align":"center"}).hide(axis="index").to_html(escape=False), unsafe_allow_html=True)
    if show_wc:
        st.sidebar.markdown("---"); st.sidebar.markdown("### ☁️ Word Cloud (Top 500)")
        wc_dynamic = st.sidebar.checkbox("Wordcloud Dinamis (mengikuti filter)", value=False, key="soc_wc_dynamic")
        if wc_dynamic:
            st.sidebar.caption("Mode: Dinamis (berdasarkan data terfilter)"); source_df = base
        else:
            st.sidebar.caption("Mode: Statis (berdasarkan data utama)"); source_df = df
        
        if not source_df.empty:
            wc_data = word_freq(source_df["content"].astype(str), 500)
            if wc_data:
                wc_df = pd.DataFrame(wc_data, columns=["Kata", "Jumlah"])
                st.sidebar.dataframe(wc_df, use_container_width=True)
            else:
                st.sidebar.write("Tidak ada kata untuk ditampilkan.")
        else:
            st.sidebar.write("Tidak ada kata untuk ditampilkan.")


# ---------- ENTRY POINT --------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Nolimit Dashboard Topic Summary")

st.sidebar.title("Pengaturan")
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("API Key Gemini ditemukan!", icon="✅")
except (FileNotFoundError, KeyError):
    st.sidebar.warning("Masukkan API Key di bawah untuk fitur AI.")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", key="api_key_input")

model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        st.sidebar.error("API Key tidak valid.")

st.sidebar.markdown("---")
st.markdown("### 📁 Pilih sumber data")
mode = st.radio("Input via:", ["Upload File", "Link Download"], horizontal=True)
data_source = None

if mode == "Upload File":
    up = st.file_uploader("Unggah file (.csv atau .zip)", type=["csv", "zip"])
    if up:
        data_source = up
else:
    url = st.text_input("URL ZIP")
    if st.button("Proceed") and url:
        tmp = "/tmp/data.zip"
        try:
            with st.spinner(f"Mengunduh dari {url}..."):
                urllib.request.urlretrieve(url, tmp)
            data_source = tmp
        except Exception as e:
            st.error(f"❌ Gagal unduh: {e}")

if "df" not in st.session_state:
    st.session_state.df = None

if data_source:
    with st.spinner("📖 Membaca data …"):
        df_all = pd.DataFrame()
        try:
            file_name = data_source.name if hasattr(data_source, 'name') else data_source
            
            if file_name.lower().endswith(".zip"):
                df_all = extract_csv_from_zip(data_source)
            elif file_name.lower().endswith(".csv"):
                df_all = pd.read_csv(data_source, delimiter=";", quotechar='"',
                                     engine="python", on_bad_lines="skip", dtype=str)
            
            if not df_all.empty:
                st.session_state.df = df_all
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
            st.session_state.df = None

# ---------- MAIN LOGIC ----------
# PERBAIKAN: Mengganti "is not in None" dengan "is not None"
if st.session_state.df is not None:
    df_raw = st.session_state.df.copy() # Gunakan .copy() untuk keamanan
    cols_lower = {c.lower() for c in df_raw.columns}
    
    is_onm = {"title", "source_name", "body"}.issubset(cols_lower)
    is_sosmed = {"content", "post_type", "final_sentiment"}.issubset(cols_lower)

    # Menyamakan nama kolom menjadi huruf kecil untuk konsistensi
    df_raw.columns = [col.lower() for col in df_raw.columns]

    if is_onm:
        st.sidebar.success("Mode: Online News Media (ONM)", icon="📰")
        run_onm(df_raw, model)

    elif is_sosmed:
        st.sidebar.success("Mode: Social Media (Sosmed)", icon="�")
        run_sosmed(df_raw, model)
    else:
        st.error("❌ Struktur kolom tidak cocok (ONM/Sosmed). Pastikan kolom wajib ada.")
        st.write("Kolom wajib ONM: `title`, `source_name`, `body`.")
        st.write("Kolom wajib Sosmed: `content`, `post_type`, `final_sentiment`.")
        st.write("Kolom yang terdeteksi:", list(cols_lower))
else:
    st.info("Unggah atau tautkan file data (.csv atau .zip) untuk memulai analisis.")
�
