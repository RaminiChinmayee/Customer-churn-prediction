import streamlit as st
import psutil
import os
import time

st.title("📡 Model Monitoring")

cpu = psutil.cpu_percent()

memory = psutil.virtual_memory()

disk = psutil.disk_usage("/")

col1, col2, col3 = st.columns(3)

col1.metric("CPU Usage", f"{cpu}%")

col2.metric("Memory Usage", f"{memory.percent}%")

col3.metric("Disk Usage", f"{disk.percent}%")

st.subheader("System Information")

st.write("Available Memory (GB)", round(memory.available / 1024**3, 2))

st.write("Total Memory (GB)", round(memory.total / 1024**3, 2))

st.write("Process ID", os.getpid())

st.write("Current Time", time.ctime())