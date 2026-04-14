"""
------------------------------------------------------------
Tutorial Utility: Streamlit Environment Sanity Check
Author: Dr. Sailesh Conjeti
Course: Generative and Agentic AI: Foundations, Frameworks and Applications
------------------------------------------------------------
Purpose:
Provide the smallest possible Streamlit app to confirm that Streamlit
is installed and can launch correctly before running larger demos.

What Students Will Learn:
- How a minimal Streamlit script is structured
- How to verify UI tooling independently of model logic

Prerequisites:
- Python environment with project requirements installed

How to Run:
streamlit run test_streamlit.py

Expected Behavior / Output:
- A browser page opens showing a title and a confirmation sentence

Key Concepts Covered:
- Basic Streamlit rendering flow
- Environment sanity testing for classroom setup
"""

import streamlit as st

# Minimal UI elements for quick environment validation.
st.title("Streamlit test")
st.write("If you can see this, Streamlit is working.")
