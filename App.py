import streamlit as st

st.set_page_config(page_title="Smart Classroom Attendance", layout="wide")

st.title("Smart Classroom Attendance System")
st.write("""
Use the sidebar to navigate:
- **Register Students** — add new students with face photos
- **Live Attendance** — run a session and detect/mark attendance from video
- **Analytics** — view attendance trends, rates, and frequent absentees
""")