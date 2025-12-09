import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation App",
    layout="wide",
    page_icon="📊"
)

# -----------------------------
# LOGIN SYSTEM
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
# SIDEBAR NAVIGATION
# -----------------------------
menu = st.sidebar.selectbox(
    "📌 Navigation",
    ["Dashboard", "Upload Data", "Segmentation Results", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.write("Developed by: **Shravanthi** 👩‍🎓")
st.sidebar.write("7th Sem – Major Project")

# -----------------------------
# CACHE LOADED DATA
# -----------------------------
@st.cache_data
def load_file(file):
    try:
        return pd.read_excel(file)
    except:
        return pd.read_csv(file)


# -----------------------------
# DASHBOARD PAGE
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
        ✔ Apply **K-Means Clustering**  
        ✔ Provide charts, clusters & insights  
        ✔ Export results for reporting  
        """)

    st.subheader("📌 Key Features")
    st.info("""
    - Machine Learning (KMeans)  
    - RFM (Recency, Frequency, Monetary)  
    - Interactive Charts (Plotly)  
    - Cluster-wise Insights  
    - Professional UI Structure  
    """)

# -----------------------------
# UPLOAD + PROCESSING PAGE
# -----------------------------
if menu == "Upload Data":
    st.title("📤 Upload Your Customer Data")

    uploaded = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

    if uploaded:
        df = load_file(uploaded)
        st.success("File uploaded successfully!")
        st.write("### Preview of data:")
        st.dataframe(df.head())

        if st.button("Process Segmentation"):
            try:
                # Auto-detect columns
                cols = df.columns.tolist()
                cust_col = next((c for c in cols if "customer" in c.lower()), cols[0])
                date_col = next((c for c in cols if "date" in c.lower()), None)
                amt_col = next((c for c in cols if "amount" in c.lower() or "price" in c.lower()), None)

                df["order_date"] = pd.to_datetime(df[date_col], errors="coerce")
                df["amount"] = pd.to_numeric(df[amt_col], errors="coerce")

                # Clean
                trans = df[[cust_col, "order_date", "amount"]].dropna()

                snapshot = trans["order_date"].max() + timedelta(days=1)

                rfm = trans.groupby(cust_col).agg(
                    recency=("order_date", lambda x: (snapshot - x.max()).days),
                    frequency=("order_date", "count"),
                    monetary=("amount", "sum")
                ).reset_index()

                rfm["avg_value"] = (rfm["monetary"] / rfm["frequency"]).round(2)

                # Log transform
                for c in ["monetary", "frequency", "avg_value"]:
                    rfm[c+"_log"] = np.log1p(rfm[c])

                features = ["recency", "frequency_log", "monetary_log", "avg_value_log"]

                scaler = StandardScaler()
                X = scaler.fit_transform(rfm[features])

                kmeans = KMeans(n_clusters=4, random_state=42)
                rfm["cluster"] = kmeans.fit_predict(X)

                st.session_state["rfm"] = rfm
                st.success("Segmentation completed!")

            except Exception as e:
                st.error(f"Processing failed: {e}")

# -----------------------------
# SEGMENTATION RESULTS PAGE
# -----------------------------
if menu == "Segmentation Results":
    st.title("📈 Segmentation Results")

    if "rfm" not in st.session_state:
        st.warning("Please upload and process data from the 'Upload Data' section.")
        st.stop()

    rfm = st.session_state["rfm"]

    st.write("### 🧩 Cluster Summary:")
    st.dataframe(rfm.groupby("cluster").mean()[["recency", "frequency", "monetary"]])

    # -------------------------
    # MATPLOTLIB BAR CHART
    # -------------------------
    st.write("### 📊 Cluster Size Chart (Matplotlib)")
    fig, ax = plt.subplots()
    rfm["cluster"].value_counts().sort_index().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # -------------------------
    # PLOTLY SCATTER
    # -------------------------
    st.write("### 🎯 Customer Distribution (Plotly)")
    fig2 = px.scatter(
        rfm,
        x="recency",
        y="monetary",
        color="cluster",
        size="frequency",
        title="Cluster Visualization"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # DOWNLOAD RESULTS
    # -------------------------
    st.download_button(
        label="📥 Download Clustered Data",
        data=rfm.to_csv(index=False),
        file_name="segmented_customers.csv",
        mime="text/csv"
    )

# -----------------------------
# ABOUT PROJECT
# -----------------------------
if menu == "About Project":
    st.title("ℹ️ About This Project")

    st.write("""
    ### **Customer Segmentation Using Unsupervised Machine Learning**

    This project helps businesses understand their customers using:

    ⭐ **RFM Model**  
    ⭐ **K-Means Clustering**  
    ⭐ **Data Visualization**  
    ⭐ **Interactive Web Application**  

    This is suitable for **Major Project / Engineering Project / B.E Final Year**.

    **Developed by:** *Shravanthi (7th Sem CSE)*
    """)