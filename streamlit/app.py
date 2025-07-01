import streamlit as st

st.set_page_config(
    page_title="Portfolio", 
    page_icon=":bar_chart:", 
    layout="wide"
)

st.title("Portfolio Dashboard")
st.markdown("""
    Welcome to my portfolio dashboard! Here you can find various projects and achievements.
    
    - **LeetCode Solutions**: Explore my solutions to LeetCode problems.
    - **Data Science Projects**: Check out my data science projects.
    - **Machine Learning Models**: View my machine learning models and their performance.
""")

with st.sidebar:
    st.header("Navigation")
    st.markdown("""
        - [LeetCode Solutions](#leetcode-solutions)
        - [Data Science Projects](#data-science-projects)
        - [Machine Learning Models](#machine-learning-models)
    """)
