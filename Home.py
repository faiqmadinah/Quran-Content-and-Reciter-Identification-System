import streamlit as st

# Page Config
st.set_page_config(
    page_title="Quran App",
    layout="centered"
)

# Title
st.title("Quran Reciter and Content Identification Application")

# Intro text
st.markdown("""
**This application offers two main functionalities:**

* **Reciter Identification**: Upload a video of a recitation, and the system will identify the reciter.
* **Content Prediction**: Enter the Quranic Verse and the system will predict Juz number, Juz name Surah number, and Surah name.

---

💡 *Tip:* Use the *sidebar* on the left to navigate between pages.
""")
