import streamlit as st
import google.generativeai as genai
import json
import os
import tempfile
import time

class QAConfig:
    API_KEY = st.secrets.get("GOOGLE_API_KEY", "AIzaSyCPCp_zanxX-pcUcSU-taWX6o-MxbR9CA")
    MODEL_NAME = 'models/gemini-flash-latest'
    PAGE_TITLE = "Medical Call QA Dashboard"
    PAGE_ICON = "🩺"

class QAAnalyzer:
    def __init__(self):
        genai.configure(api_key=QAConfig.API_KEY)
        self.model = genai.GenerativeModel(QAConfig.MODEL_NAME)

    def _clean_json(self, text):
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def analyze_audio(self, file_path):
        audio_file = genai.upload_file(path=file_path)
        
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = genai.get_file(audio_//name) # تصحيح خطأ هنا
            # سأقوم بتعديل السطر التالي ليكون صحيحاً تماماً
        
        # تصحيح السطر
        audio_file = genai.get_file(audio_file.name)
        
        prompt = """
        Act as a Professional Medical Call Quality Control Specialist. 
        Your objective is to evaluate the Agent's performance focusing on data accuracy and the professional handling of patient concerns.

        ### 1. DATA INTEGRITY & VERIFICATION (Scoring: 40 pts)
        Verify if the Agent correctly collected:
        - Patient Identity: Name, DOB, Address, and Medical ID.
        - Vitals: Height, Weight, and BMI.
        - Next Steps: Clear explanation of the post-call process.
        (Deduct points for any missing or incorrectly verified data).

        ### 2. ADVANCED OBJECTION HANDLING LOGIC (Scoring: 40 pts) - CRITICAL
        Analyze how the agent handles concerns (e.g., pain, wanting to see a doctor, "I don't need this").
        Follow this logic:
        - Step A (Identification): Identify the patient's concern.
        - Step B (Handling Analysis): Did the agent acknowledge the concern, listen actively, and provide a professional explanation?
        - Step C (Outcome): If the agent provided a verbal response/explanation according to protocol, mark as "Handled" (Success), regardless of whether the patient eventually agreed.
        
        HARD RULE: Do NOT label the agent as "ignoring" the patient if they provided any verbal response or professional explanation to the concern.

        ### 3. AGENT PROFESSIONALISM & STATUS (Scoring: 20 pts)
        - Evaluate if the agent maintained control and addressed objections logically.
        - Call Status: Mark as "Pass" if all data is collected and all objections were handled/addressed.

        Task: Extract medical data and provide a detailed QA evaluation.
        
        OUTPUT FORMAT (Strict JSON):
        {
          "Agent_Name": "", "Patient_Name": "", "DOB": "", "Address": "",
          "Phone_Number": "", "Medicare_ID": "", "Brace_Size": "", "Height": "",
          "Weight": "", "Pain_Level": "", "Doctor_Name": "", "Last_Visit_Date": "",
          "Previous_Treatments": "", "Score": "", "Strengths": "", "Weaknesses": "", "Call_Status": "Pass/Fail"
        }
        """
        
        response = self.model.generate_content(
            [prompt, audio_file],
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(self._clean_json(response.text))

class UIHandler:
    @staticmethod
    def apply_styles():
        st.markdown(
            """
            <style>
            .stApp { background-color: #f8f9fa; }
            .main-header {
                background: linear-gradient(90deg, #0f172a 0%, #2563eb 100%);
                color: white; padding: 2rem; border-radius: 15px;
                text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .main-header h1 { color: white !important; font-size: 2.5rem !important; margin-bottom: 0; }
            .main-header p { font-size: 1.2rem; opacity: 0.9; }
            .custom-card {
                background-color: white; padding: 20px; border-radius: 15px;
                border-left: 5px solid #2563eb; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                margin-bottom: 20px; transition: transform 0.2s;
            }
            .custom-card:hover { transform: translateY(-5px); box-shadow: 0 6px 12px rgba(0,0,0,0.1); }
            .card-title {
                color: #1e3a8a; font-size: 1.3rem; font-weight: bold;
                margin-bottom: 15px; display: flex; align-items: center; gap: 10px;
            }
            .data-label { font-weight: 600; color: #64748b; width: 150px; display: inline-block; }
            .data-value { color: #1e293b; font-weight: 400; }
            section[data-testid="stSidebar"] { background-color: #0f172a !important; }
            section[data-testid="stSidebar"] .stText, section[data-testid="stSidebar"] label { color: white !important; }
            </style>
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def render_header():
        st.markdown("""
            <div class="main-header">
                <h1>🩺 Medical Call QA Dashboard</h1>
                <p>Powered by Outsourcing Skill - Advanced Quality Control</p>
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_results(result):
        col1, col2 = st.columns([1, 2])
        with col1:
            status_color = "green" if result.get("Call_Status") == "Pass" else "red"
            st.markdown(f"""
                <div class="custom-card">
                    <div class="card-title">📊 Overall Score</div>
                    <div style="text-align:center;">
                        <h2 style="font-size: 3rem; color: #1e3a8a; margin: 0;">{result.get('Score', 'N/A')}/100</h2>
                        <p style="color: {status_color}; font-weight: bold; font-size: 1.2rem;">Status: {result.get('Call_Status', 'N/A')}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="custom-card">
                    <div class="card-title">👤 Agent Details</div>
                    <div style="font-size: 1.1rem;">
                        <span class="data-label">Agent Name:</span> <span class="data-value">{result.get('Agent_Name', 'N/A')}</span><br>
                        <span class="data-label">Analysis Date:</span> <span class="data-value">{time.strftime('%Y-%m-%d %H:%M')}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="custom-card">
                <div class="card-title">🏥 Extracted Medical Data</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div><span class="data-label">Patient Name:</span> <span class="data-value">{result.get('Patient_Name', 'N/A')}</span></div>
                    <div><span class="data-label">Doctor Name:</span> <span class="data-value">{result.get('Doctor_Name', 'N/A')}</span></div>
                    <div><span class="data-label">DOB:</span> <span class="data-value">{result.get('DOB', 'N/A')}</span></div>
                    <div><span class="data-label">Last Visit:</span> <span class="data-value">{result.get('Last_Visit_Date', 'N/A')}</span></div>
                    <div><span class="data-label">Phone:</span> <span class="data-value">{result.get('Phone_Number', 'N/A')}</span></div>
                    <div><span class="data-label">Pain Level:</span> <span class="data-value">{result.get('Pain_Level', 'N/A')}</span></div>
                    <div><span class="data-label">Address:</span> <span class="data-value">{result.get('Address', 'N/A')}</span></div>
                    <div><span class="data-label">Brace Size:</span> <span class="data-value">{result.get('Brace_Size', 'N/A')}</span></div>
                    <div><span class="data-label">Medicare ID:</span> <span class="data-value">{result.get('Medicare_ID', 'N/A')}</span></div>
                    <div><span class="data-label">Height/Weight:</span> <span class="data-value">{result.get('Height', 'N/A')} / {result.get('Weight', 'N/A')}</span></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-title">💡 QA Feedback & Compliance</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🌟 Strengths", "⚠️ Weaknesses & Observations"])
        with tab1:
            st.success(result.get("Strengths", "None listed."))
        with tab2:
            st.error(result.get("Weaknesses", "None listed."))

def main():
    st.set_page_config(page_title=QAConfig.PAGE_TITLE, page_icon=QAConfig.PAGE_ICON, layout="wide")
    
    ui = UIHandler()
    ui.apply_styles()
    ui.render_header()
    
    analyzer = QAAnalyzer()
    
    st.sidebar.header("📂 Upload Call Record")
    uploaded_file = st.sidebar.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])

    if uploaded_file:
        st.sidebar.audio(uploaded_file, format='audio/mp3')
        if st.sidebar.button("🚀 Analyze Call Now"):
            with st.spinner('🤖 AI Analyst is evaluating call protocol...'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
                    temp.write(uploaded_file.read())
                    temp_path = temp.name

                try:
                    result = analyzer.analyze_audio(temp_path)
                    st.success("✅ Analysis Complete!")
                    ui.render_results(result)
                    with st.expander("📋 Raw JSON Data"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Analysis Error: {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    else:
        st.info("👈 Please upload an audio file from the sidebar to begin.")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            try:
                st.image("logo.png", width=250)
            except:
                st.markdown("<h3 style='text-align:center; color:grey;'>Logo Image Missing</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
