import streamlit as st

st.set_page_config(page_title="PFDI Test", layout="wide")

st.title("Punjab Farmer Distress App")
st.write("If you can see this, deployment is working.")

st.header("Test Form")

name = st.text_input("Farmer name", "Test Farmer")
score = st.slider("Sample distress score", 0.0, 1.0, 0.5)

if st.button("Test Button"):
    st.success(f"{name} -> score = {score}")