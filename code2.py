import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
import io

# Optional PDF export (FPDF).
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

st.set_page_config(page_title="Customer Segmentation App", layout="wide", page_icon="📊")

# -----------------------------
# Simple login
# -----------------------------
def login():
    st.title("🔐 Login")
    st.markdown("### Welcome to the Customer Segmentation System")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and pwd == "123":
            st.session_state["logged"] = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

if "logged" not in st.session_state:
    st.session_state["logged"] = False
if not st.session_state["logged"]:
    login()
    st.stop()

# -----------------------------
# Sidebar & navigation
# -----------------------------
menu = st.sidebar.selectbox("📌 Navigation", ["Dashboard", "Upload Data", "Segmentation Results", "Algorithm Comparison", "About Project"])
st.sidebar.markdown("---")
st.sidebar.write("Developed by: *Shravanthi* 👩‍🎓 and  *Shreedevi S Haller* 👩‍🎓")
st.sidebar.write("7th Sem – Major Project")

# -----------------------------
# File loader cache
# -----------------------------
@st.cache_data
def load_file(file):
    try:
        return pd.read_excel(file)
    except Exception:
        try:
            return pd.read_csv(file)
        except Exception as e:
            raise e

# -----------------------------
# Helper functions
# -----------------------------
def clean_data_basic(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    return df

def remove_outliers_iqr(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            Q1 = df[c].quantile(0.25)
            Q3 = df[c].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[c] >= lower) & (df[c] <= upper)]
    return df

def rfm_scores(df, r_col="recency", f_col="frequency", m_col="monetary"):
    df = df.copy()
    try:
        df['R_score'] = pd.qcut(df[r_col], 5, labels=[5,4,3,2,1]).astype(int)
    except Exception:
        df['R_score'] = pd.cut(df[r_col].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
    try:
        df['F_score'] = pd.qcut(df[f_col].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    except Exception:
        df['F_score'] = pd.cut(df[f_col].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    try:
        df['M_score'] = pd.qcut(df[m_col], 5, labels=[1,2,3,4,5]).astype(int)
    except Exception:
        df['M_score'] = pd.cut(df[m_col].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    df['RFM_Score'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)
    df['RFM_total'] = df['R_score'] + df['F_score'] + df['M_score']
    return df

def label_segments_from_rfm(df):
    df = df.copy()
    def label(x):
        if x >= 13: return "Champion"
        if 10 <= x <= 12: return "Loyal"
        if 8 <= x <= 9: return "Potential"
        if 5 <= x <= 7: return "Needs Attention"
        return "At Risk"
    df['segment_label'] = df['RFM_total'].apply(label)
    return df

def cluster_insights(df, cluster_col="cluster"):
    insights = {}
    if cluster_col not in df.columns:
        return insights
    groups = df.groupby(cluster_col)
    for cid, g in groups:
        rec = g['recency'].mean() if 'recency' in g.columns else np.nan
        freq = g['frequency'].mean() if 'frequency' in g.columns else np.nan
        mon = g['monetary'].mean() if 'monetary' in g.columns else np.nan
        avgval = g['avg_value'].mean() if 'avg_value' in g.columns else (mon/(freq+1e-9) if (not np.isnan(mon) and not np.isnan(freq)) else np.nan)
        size = len(g)
        category = []
        if 'monetary' in df.columns:
            if mon >= df['monetary'].quantile(0.75): category.append("High monetary")
            elif mon <= df['monetary'].quantile(0.25): category.append("Low monetary")
        if 'recency' in df.columns:
            if rec <= df['recency'].quantile(0.25): category.append("Recent buyers")
            elif rec >= df['recency'].quantile(0.75): category.append("Inactive/old")
        if 'frequency' in df.columns:
            if freq >= df['frequency'].quantile(0.75): category.append("Frequent buyers")
            else: category.append("Infrequent")
        insights[cid] = {
            "size": int(size),
            "avg_recency": float(round(rec,2)) if not np.isnan(rec) else None,
            "avg_frequency": float(round(freq,2)) if not np.isnan(freq) else None,
            "avg_monetary": float(round(mon,2)) if not np.isnan(mon) else None,
            "avg_value": float(round(avgval,2)) if not np.isnan(avgval) else None,
            "categories": ", ".join(category) if category else ""
        }
    return insights

def make_pie_chart(df, cluster_col="cluster"):
    if cluster_col not in df.columns:
        return px.pie(title="No cluster data")
    counts = df[cluster_col].value_counts().reset_index()
    counts.columns = ['cluster', 'count']
    fig = px.pie(counts, names='cluster', values='count', title="Cluster Distribution")
    return fig

def try_pdf_export(rfm, insights, filename="segmentation_report.pdf"):
    if not FPDF_AVAILABLE:
        return None, "FPDF not available. Install `fpdf` to enable PDF export (pip install fpdf)."
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Customer Segmentation Report", ln=1, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, f"Total customers: {len(rfm)}", ln=1)
    pdf.ln(4)
    for cid, info in insights.items():
        pdf.cell(0, 6, f"Cluster {cid} - size: {info.get('size', 'N/A')}", ln=1)
        pdf.cell(0, 6, f"  Avg Recency: {info.get('avg_recency', 'N/A')} days", ln=1)
        pdf.cell(0, 6, f"  Avg Frequency: {info.get('avg_frequency', 'N/A')}", ln=1)
        pdf.cell(0, 6, f"  Avg Monetary: {info.get('avg_monetary', 'N/A')}", ln=1)
        pdf.cell(0, 6, f"  Categories: {info.get('categories', '')}", ln=1)
        pdf.ln(2)
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    bio = io.BytesIO(pdf_bytes); bio.seek(0)
    return bio, None

# -----------------------------
# Dashboard page
# -----------------------------
if menu == "Dashboard":
    st.title("📊 Customer Segmentation Dashboard")
    st.markdown("### Enhance your business insights using ML-powered segmentation.")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=230)
    with col2:
        st.write("""
        ### What this system does:
        ✔ Upload customer transaction data  
        ✔ Automatically clean & calculate RFM  
        ✔ Apply *K-Means Clustering*  
        ✔ Provide charts, clusters & insights  
        ✔ Export results for reporting  
        """)
    st.subheader("📌 Key Features")
    st.info("""
    - Machine Learning (KMeans & Agglomerative)  
    - RFM (Recency, Frequency, Monetary)  
    - Algorithm Comparison  
    - Cluster Quality Metrics  
    - Interactive Charts (Plotly)  
    - Professional UI Structure  
    """)

# -----------------------------
# Upload & process data
# -----------------------------
if menu == "Upload Data":
    st.title("📤 Upload Your Customer Data")
    uploaded = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])
    if uploaded:
        try:
            df = load_file(uploaded)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.stop()
        st.success("File uploaded successfully!")
        st.write("### Preview of data:")
        st.dataframe(df.head())

        # Auto-detect columns (no manual selection UI)
        cols = df.columns.tolist()
        cust_col = next((c for c in cols if "customer" in c.lower() or "cust" in c.lower() or c.lower() == "id"), cols[0])
        date_col = next((c for c in cols if "date" in c.lower()), None)
        amt_col = next((c for c in cols if "amount" in c.lower() or "price" in c.lower() or "amt" in c.lower() or "value" in c.lower()), None)

        if st.button("Process Segmentation"):
            try:
                # parse with detected columns
                if date_col is None or amt_col is None:
                    st.error("Could not detect date or amount column automatically. Please ensure your file has a date and an amount column.")
                    st.stop()

                df["order_date"] = pd.to_datetime(df[date_col], errors="coerce")
                df["amount"] = pd.to_numeric(df[amt_col], errors="coerce")

                trans = df[[cust_col, "order_date", "amount"]].dropna(subset=[cust_col, "order_date", "amount"])
                if trans.empty:
                    st.error("No valid transactions after parsing date/amount. Check input columns and data.")
                    st.stop()

                snapshot = trans["order_date"].max() + timedelta(days=1)

                rfm = trans.groupby(cust_col).agg(
                    recency=("order_date", lambda x: (snapshot - x.max()).days),
                    frequency=("order_date", "count"),
                    monetary=("amount", "sum")
                ).reset_index()

                rfm["avg_value"] = (rfm["monetary"] / rfm["frequency"]).round(2)

                for c in ["monetary", "frequency", "avg_value"]:
                    if c in rfm.columns:
                        rfm[c + "_log"] = np.log1p(rfm[c].clip(lower=0))

                # no optional cleaning UI — apply basic cleaning by default
                rfm = clean_data_basic(rfm)

                features = [f for f in ["recency", "frequency_log", "monetary_log", "avg_value_log"] if f in rfm.columns]
                if len(features) < 2:
                    st.error("Not enough features for clustering.")
                    st.stop()

                scaler = StandardScaler()
                X = scaler.fit_transform(rfm[features])

                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                rfm["cluster"] = kmeans.fit_predict(X)

                rfm = rfm_scores(rfm, r_col="recency", f_col="frequency", m_col="monetary")
                rfm = label_segments_from_rfm(rfm)

                st.session_state["rfm"] = rfm
                st.session_state["X"] = X
                st.session_state["cust_col_name"] = cust_col
                st.success("Segmentation completed!")
            except Exception as e:
                st.error(f"Processing failed: {e}")

# -----------------------------
# Segmentation Results page
# -----------------------------
if menu == "Segmentation Results":
    st.title("📈 Segmentation Results")
    if "rfm" not in st.session_state:
        st.warning("Please upload and process data first."); st.stop()
    rfm = st.session_state["rfm"]
    id_col = st.session_state.get("cust_col_name", rfm.columns[0])
    st.write("### 📘 RFM Values for Each Customer")
    show_cols = [id_col, "recency", "frequency", "monetary", "avg_value", "R_score", "F_score", "M_score", "RFM_Score", "RFM_total", "segment_label", "cluster"]
    show_cols = [c for c in show_cols if c in rfm.columns]
    st.dataframe(rfm[show_cols].sort_values("RFM_total", ascending=False).reset_index(drop=True))

    st.write("### 🧩 Cluster Summary (KMeans)")
    agg_cols = [c for c in ["recency", "frequency", "monetary", "avg_value"] if c in rfm.columns]
    if agg_cols:
        try:
            rfm[agg_cols] = rfm[agg_cols].apply(pd.to_numeric, errors='coerce')
            summary = rfm.groupby("cluster")[agg_cols].mean(numeric_only=True).round(2)
            st.dataframe(summary)
        except Exception:
            st.info("Cluster summary unavailable due to non-numeric data.")
    else:
        st.info("No numeric columns to summarize.")

    st.write("### 📊 Cluster Distribution (KMeans)")
    st.plotly_chart(make_pie_chart(rfm, "cluster"), use_container_width=True)

    st.write("### 🎯 Customer Distribution (Plotly - KMeans)")
    x_field = "recency" if "recency" in rfm.columns else rfm.columns[0]
    y_field = "monetary" if "monetary" in rfm.columns else (rfm.columns[1] if len(rfm.columns) > 1 else rfm.columns[0])
    size_field = "frequency" if "frequency" in rfm.columns else None
    fig2 = px.scatter(rfm, x=x_field, y=y_field, color="cluster" if "cluster" in rfm.columns else None, size=size_field, hover_data=[id_col] + (["RFM_Score","segment_label"] if "RFM_Score" in rfm.columns else []), title="Cluster Visualization (KMeans)")
    st.plotly_chart(fig2, use_container_width=True)

    # removed radar chart (as requested)

    st.write("### 🔎 Cluster Insights (automatic interpretation)")
    insights = cluster_insights(rfm, cluster_col="cluster")
    if insights:
        st.dataframe(pd.DataFrame.from_dict(insights, orient='index'))
    else:
        st.info("No insights available (cluster column missing).")

    st.write("### 📥 Download Clustered Data (KMeans)")
    st.download_button(label="Download Segmented Customers CSV", data=rfm.to_csv(index=False), file_name="segmented_customers_kmeans.csv", mime="text/csv")

    if FPDF_AVAILABLE:
        if st.button("📄 Export simple PDF Report"):
            bio, err = try_pdf_export(rfm, insights)
            if err: st.error(err)
            else: st.download_button("Download PDF report", data=bio, file_name="segmentation_report.pdf", mime="application/pdf")
    else:
        st.info("PDF export not available (install fpdf).")

# -----------------------------
# Algorithm Comparison page (table + winner)
# -----------------------------
if menu == "Algorithm Comparison":
    st.title("🤖 Algorithm Comparison")

    if "rfm" not in st.session_state:
        st.warning("Please upload and process data first."); st.stop()

    rfm = st.session_state["rfm"]
    X = st.session_state["X"]

    # Re-run/fit algorithms
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    rfm["kmeans_cluster"] = kmeans.labels_

    agg = AgglomerativeClustering(n_clusters=4).fit(X)
    rfm["agg_cluster"] = agg.labels_

    # removed DBSCAN UI settings; run with default sensible parameters
    try:
        dbs = DBSCAN(eps=0.5, min_samples=5).fit(X)
        rfm["dbscan_cluster"] = dbs.labels_
    except Exception:
        rfm["dbscan_cluster"] = -1

    def safe_metrics(X, labels):
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels != -1] if -1 in unique_labels else unique_labels
        if len(valid_labels) <= 1:
            return {"Silhouette": np.nan, "Calinski": np.nan, "Davies": np.nan}
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = np.nan
        try:
            ch = calinski_harabasz_score(X, labels)
        except Exception:
            ch = np.nan
        try:
            db = davies_bouldin_score(X, labels)
        except Exception:
            db = np.nan
        return {"Silhouette": sil, "Calinski": ch, "Davies": db}

    km_scores = safe_metrics(X, rfm["kmeans_cluster"])
    agg_scores = safe_metrics(X, rfm["agg_cluster"])
    dbs_scores = safe_metrics(X, rfm["dbscan_cluster"])

    comp_df = pd.DataFrame({
        "KMeans": km_scores,
        "Agglomerative": agg_scores,
        "DBSCAN": dbs_scores
    }).T

    # rename and format
    comp_df = comp_df.rename(columns={
        "Silhouette": "Silhouette Score",
        "Calinski": "Calinski-Harabasz",
        "Davies": "Davies-Bouldin (lower=better)"
    })
    # round numeric values for display
    comp_df_display = comp_df.copy()
    comp_df_display = comp_df_display.applymap(lambda v: round(v,4) if pd.notna(v) else np.nan)

    st.write("### 📊 Algorithm Metrics Comparison")
    # highlight best per column (note: Davies lower=better)
    def highlight_best(s, colname):
        if s.dropna().empty:
            return ['' for _ in s]
        if "Davies-Bouldin" in colname:
            best = s.idxmin()
            return ['background-color: #b7e4c7' if idx == best else '' for idx in s.index]
        else:
            best = s.idxmax()
            return ['background-color: #b7e4c7' if idx == best else '' for idx in s.index]

    # Show table and highlight best per metric
    styled = comp_df_display.style
    for col in comp_df_display.columns:
        styled = styled.apply(lambda s, colname=col: highlight_best(s, colname), axis=0)
    st.dataframe(styled)

    # Decide winners per metric
    def pick_best(metric_col):
        vals = comp_df[metric_col]
        # ignore NaNs
        vals = vals.dropna()
        if vals.empty:
            return None
        if "Davies" in metric_col:
            return vals.idxmin()
        else:
            return vals.idxmax()

    sil_best = pick_best("Silhouette Score")
    ch_best = pick_best("Calinski-Harabasz")
    db_best = pick_best("Davies-Bouldin (lower=better)")

    # Voting to pick overall winner
    votes = {"KMeans": 0, "Agglomerative": 0, "DBSCAN": 0}
    for w in [sil_best, ch_best, db_best]:
        if w in votes:
            votes[w] += 1
    max_votes = max(votes.values())
    candidates = [alg for alg, v in votes.items() if v == max_votes]

    if len(candidates) == 1:
        overall = candidates[0]
    else:
        # tie-breaker: silhouette -> calinski -> davies
        def get_sil(a): return comp_df.loc[a, "Silhouette Score"]
        sil_vals = {a: get_sil(a) for a in candidates}
        sil_vals = {k:v for k,v in sil_vals.items() if pd.notna(v)}
        if sil_vals:
            overall = max(sil_vals, key=sil_vals.get)
        else:
            def get_ch(a): return comp_df.loc[a, "Calinski-Harabasz"]
            ch_vals = {a: get_ch(a) for a in candidates}
            ch_vals = {k:v for k,v in ch_vals.items() if pd.notna(v)}
            if ch_vals:
                overall = max(ch_vals, key=ch_vals.get)
            else:
                def get_db(a): return comp_df.loc[a, "Davies-Bouldin (lower=better)"]
                db_vals = {a: get_db(a) for a in candidates}
                db_vals = {k:v for k,v in db_vals.items() if pd.notna(v)}
                if db_vals:
                    overall = min(db_vals, key=db_vals.get)
                else:
                    overall = candidates[0]

    # Display winners and overall
    st.write("")
    st.success(f"Best per metric — Silhouette: {sil_best or 'N/A'}, Calinski-Harabasz: {ch_best or 'N/A'}, Davies-Bouldin: {db_best or 'N/A'}")
    st.success(f"Overall best algorithm: {overall}")

# -----------------------------
# About page
# -----------------------------
if menu == "About Project":
    st.title("ℹ About This Project")
    st.write("""
    ### *Customer Segmentation Using Unsupervised Machine Learning*

    This project helps businesses understand their customers using:

    ⭐ **RFM Model**  
    ⭐ **K-Means & Agglomerative Clustering**  
    ⭐ **DBSCAN (optional, tunable)**  
    ⭐ **RFM Scoring & Segment Labels**  
    ⭐ **Cluster Insights & Radar Charts**  
    ⭐ **PDF Export (optional, requires fpdf)**

    Ideal for Major Project / Engineering Project / B.E Final Year.

    *Developed by:* **Shravanthi (7th Sem CSE)** **Shreedevi S Haller (7th Sem CSE)**
    """)
