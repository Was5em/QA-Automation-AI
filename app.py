import streamlit as st
import google.generativeai as genai
import json
import os
import tempfile
import time

class QAConfig:
    API_KEY = st.secrets.get("GOOGLE_API_KEY", "AIzaSyDjOP3Ps9lsLAeEp5bgexGMAn7AJqn04Ek")
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
            audio_file = genai.get_file(audio_file.name)
        
        prompt = """
        Act as a Balanced Medical Call Quality Assurance Specialist. 
        Your goal is to provide a fair and objective evaluation of the agent's performance.

        ### SCORING SYSTEM (Total 100 pts):
        1. Data Accuracy (40 pts): Verify Patient Identity, Vitals, and Next Steps.
        2. Objection Handling (40 pts): Evaluate if the agent acknowledged concerns and provided professional explanations.
        3. Professionalism (20 pts): Call control and clear closing.

        HARD RULE: Do NOT label the agent as "ignoring" the patient if they provided any verbal response or professional explanation to the concern.

        Task: Extract medical data and provide a balanced QA evaluation.
        
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
        
        # تحويل النص إلى JSON
        raw_result = json.loads(self._clean_json(response.text))
        
        # حل مشكلة 'list object has no attribute get'
        # إذا كانت النتيجة قائمة، نأخذ العنصر الأول منها ليكون Dictionary
        if isinstance(raw_result, list):
            return raw_result[0] if len(raw_result) > 0 else {}
            
        return raw_result

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
        if not result:
            st.error("No data received from AI.")
            return

        col1, col2 = st.columns([1, 2])
        with col1:
            status_color = "green" if result.get("Call_Status") == "Pass" else "red"
            st.markdown(f"""
                <div class="custom-card">
                    <div class="card-title">📊 Overall Score</div>
                    <div style="text-align:cente
