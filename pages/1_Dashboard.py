import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json


st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)


# ----------------------------
# Load Data
# ----------------------------

df = pd.read_csv(
    "data/telecom.csv"
)


# ----------------------------
# Header
# ----------------------------

st.title("📊 Customer Churn Intelligence Dashboard")

st.markdown(
"""
Enterprise-level analytics dashboard for predicting,
monitoring and reducing customer churn.
"""
)


# ----------------------------
# KPI SECTION
# ----------------------------


total_customers = len(df)

churn_customers = len(
    df[df["Churn"]=="Yes"]
)

churn_rate = (
    churn_customers /
    total_customers
)*100


monthly_revenue = (
    df["MonthlyCharges"]
    .sum()
)


lost_revenue = (
    df[df["Churn"]=="Yes"]
    ["MonthlyCharges"]
    .sum()
)


avg_tenure = (
    df["tenure"]
    .mean()
)



c1,c2,c3,c4,c5 = st.columns(5)


c1.metric(
    "👥 Customers",
    f"{total_customers:,}"
)


c2.metric(
    "🚨 Churn Rate",
    f"{churn_rate:.2f}%"
)


c3.metric(
    "💰 Revenue",
    f"£{monthly_revenue:,.0f}"
)


c4.metric(
    "📉 Revenue Lost",
    f"£{lost_revenue:,.0f}"
)


c5.metric(
    "⏳ Avg Tenure",
    f"{avg_tenure:.1f} months"
)



st.divider()



# ----------------------------
# CHURN OVERVIEW
# ----------------------------


col1,col2 = st.columns(2)


with col1:

    fig = px.pie(
        df,
        names="Churn",
        hole=0.5,
        title="Customer Churn Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    contract = (
        df.groupby(
            ["Contract","Churn"]
        )
        .size()
        .reset_index(
            name="Count"
        )
    )


    fig = px.bar(
        contract,
        x="Contract",
        y="Count",
        color="Churn",
        barmode="group",
        title="Churn by Contract Type"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



# ----------------------------
# CUSTOMER BEHAVIOUR
# ----------------------------


st.subheader(
    "👤 Customer Behaviour Analysis"
)


col3,col4 = st.columns(2)


with col3:

    fig = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=20,
        title="Tenure vs Churn"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



with col4:

    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        title="Monthly Charges Impact"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



# ----------------------------
# PAYMENT ANALYSIS
# ----------------------------


st.subheader(
    "💳 Payment Risk Analysis"
)



payment = (
    df.groupby(
        ["PaymentMethod","Churn"]
    )
    .size()
    .reset_index(
        name="Customers"
    )
)


fig = px.bar(
    payment,
    x="PaymentMethod",
    y="Customers",
    color="Churn",
    title="Payment Method vs Churn"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



# ----------------------------
# INTERNET SERVICE ANALYSIS
# ----------------------------


st.subheader(
    "🌐 Internet Service Analysis"
)



internet = (
    df.groupby(
        ["InternetService","Churn"]
    )
    .size()
    .reset_index(
        name="Count"
    )
)



fig = px.sunburst(
    internet,
    path=[
        "InternetService",
        "Churn"
    ],
    values="Count",
    title="Internet Service Churn"
)



st.plotly_chart(
    fig,
    use_container_width=True
)




# ----------------------------
# RISK SEGMENTATION
# ----------------------------


st.subheader(
    "🚦 Customer Risk Segmentation"
)



def risk(row):

    if row["tenure"] < 12:
        return "High Risk"

    elif row["MonthlyCharges"] > 80:
        return "Medium Risk"

    else:
        return "Low Risk"



df["Risk"] = df.apply(
    risk,
    axis=1
)


risk_count = (
    df["Risk"]
    .value_counts()
    .reset_index()
)


risk_count.columns=[
    "Risk",
    "Customers"
]


fig = px.bar(
    risk_count,
    x="Risk",
    y="Customers",
    color="Risk",
    title="Customer Risk Groups"
)


st.plotly_chart(
    fig,
    use_container_width=True
)



# ----------------------------
# MODEL PERFORMANCE
# ----------------------------


st.subheader(
    "🤖 Model Performance"
)



try:

    with open(
        "models/model_metrics.json"
    ) as f:

        metrics=json.load(f)


    metric_df=pd.DataFrame(metrics).T


    best_model = (
        metric_df["ROC_AUC"]
        .idxmax()
    )


    a,b,c = st.columns(3)


    a.metric(
        "Best Model",
        best_model
    )


    b.metric(
        "ROC-AUC",
        metric_df.loc[
            best_model,
            "ROC_AUC"
        ]
    )


    c.metric(
        "F1 Score",
        metric_df.loc[
            best_model,
            "F1"
        ]
    )

except:

    st.warning(
        "Train model to view metrics"
    )



# ----------------------------
# DOWNLOAD DATA
# ----------------------------


st.subheader(
    "📥 Export Dashboard Data"
)


csv=df.to_csv(
    index=False
)


st.download_button(
    "Download Customer Report",
    csv,
    "customer_churn_report.csv",
    "text/csv"
)