import streamlit as st
import pandas as pd

st.set_page_config(page_title="Churn & Retention Offers", layout="wide")

@st.cache_data
def load():
    return pd.read_csv("scored_customers.csv")

df = load()

st.title("Customer Churn & Retention Offer Engine")
st.caption("Success rates behind the offer recommendations are assumptions, "
           "not measured results. Data is synthetic.")

# ---- Sidebar filters ----
st.sidebar.header("Filters")
all_offers = sorted(df["recommended_offer"].unique())
offers = st.sidebar.multiselect("Recommended offer", all_offers,
                                default=[o for o in all_offers if o != "none"])
all_drivers = sorted(d for d in df["main_driver"].unique() if d != "-")
drivers = st.sidebar.multiselect("Main churn driver", all_drivers, default=all_drivers)
min_p = st.sidebar.slider("Minimum churn probability", 0.0, 1.0, 0.5, 0.05)
sort_by = st.sidebar.selectbox("Rank by", ["expected_value", "churn_prob", "total_spend"])
top_n = st.sidebar.number_input("Rows to show", 10, 1000, 50, step=10)

f = df[
    df["recommended_offer"].isin(offers)
    & (df["main_driver"].isin(drivers) | (df["main_driver"] == "-"))
    & (df["churn_prob"] >= min_p)
]

# ---- KPIs ----
c1, c2, c3 = st.columns(3)
c1.metric("Customers matching filters", f"{len(f):,}")
c2.metric("Average churn probability", f"{f['churn_prob'].mean():.2f}" if len(f) else "-")
c3.metric("Total expected value (assumed)", f"{f['expected_value'].sum():,.0f}")

# ---- Charts ----
left, right = st.columns(2)
left.subheader("Recommended offers")
left.bar_chart(f["recommended_offer"].value_counts())
right.subheader("Churn drivers")
right.bar_chart(f["main_driver"].value_counts())

# ---- Ranked table ----
st.subheader("High-risk customers, ranked")
st.dataframe(f.sort_values(sort_by, ascending=False).head(int(top_n)),
             use_container_width=True, hide_index=True)

# ---- Customer lookup ----
st.subheader("Look up one customer")
cid = st.number_input("Customer ID", min_value=0, value=int(df["customer_id"].iloc[0]), step=1)
row = df[df["customer_id"] == cid]
if row.empty:
    st.info("Customer not found in the scored test set.")
else:
    r = row.iloc[0]
    st.write(f"**Churn probability:** {r['churn_prob']:.2f}")
    st.write(f"**Reason:** {r['reason']}")
    st.write(f"**Recommended offer:** {r['recommended_offer']} (expected value {r['expected_value']})")