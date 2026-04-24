import streamlit as st
import google.generativeai as genai
import json
import os
import tempfile
import time

# ==========================================
# 1. الإعدادات العامة (Configuration)
# ==========================================
class QAConfig:
    # يفضل وضع المفتاح في st.secrets عند الرفع على Streamlit Cloud
    API_KEY = st.secrets.get("GOOGLE_API_KEY", "AIzaSyDjOP3Ps9lsLAeEp5bgexGMAn7AJqn04Ek")
    MODEL_NAME = 'models/gemini-flash-latest'
    PAGE_TITLE = "Medical Call QA Dashboard"
    PAGE_ICON = "🩺"

# ==========================================
# 2. محرك التحليل (AI Analyzer)
# ==========================================
class QAAnalyzer:
    def __init__(self):
        genai.configure(api_key=QAConfig.API_KEY)
        self.model = genai.GenerativeModel(QAConfig.MODEL_NAME)

    def _clean_json(self, text):
        """تنظيف النص المستخرج من AI للتأكد من أنه JSON صالح"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def analyze_audio_final(self, file_path):
        # 1. رفع الملف إلى Google Gemini
        audio_file = genai.upload_file(path=file_path)
        
        # 2. الانتظار حتى تنتهي عملية المعالجة (Processing)
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = genai.get_file(audio_file.name)
        
        # 3. البرومبت الصارم (Strict Prompt) - تم تعديله ليكون واقعياً وغير متساهل
        prompt = """
        Act as a Strict Medical Call Quality Assurance Auditor. Your primary goal is to identify risks, errors, and gaps in call handling. 
        Do NOT be overly lenient. Patience is a soft skill, but Compliance and Accuracy are hard requirements.

        ### CRITICAL EVALUATION CRITERIA:
        1. Data Accuracy & Compliance (40 pts): 
           - Did the agent verify all data without unnecessary repetition?
           - Did the agent clearly state the purpose of the call?
           - RED FLAG: Continuing the pitch after the patient clearly refused or expressed distress.
        
        2. Call Control & Communication (40 pts):
           - Did the agent manage the patient's emotions or just "stay quiet" (passive behavior)?
           - Was the explanation of the product/service consistent and clear?
           - RED FLAG: Repeating the same questions multiple times (indicates poor listening/control).
        
        3. Professionalism & Ethics (20 pts):
           - Did the agent respect the patient's mental state and confusion?
           - Was the closing professional and compliant?

        ### SCORING GUIDELINES (BE CRITICAL):
        - 100/100: Only for a flawless call with ZERO errors.
        - 80-90: Good call but with minor phrasing issues.
        - 60-79: Moderate issues in call control, repetition, or communication.
        - Below 60: Major compliance risks, poor patient handling, or ignoring refusals.

        ### TASK:
        Analyze the audio critically. First, identify all the mistakes and compliance risks. 
        Then, determine if the agent's patience actually solved the problem or if it was just "passive" behavior that didn't lead to a quality result.

        OUTPUT FORMAT (Strict JSON object, NOT a list):
        {
          "Agent_Name": "", "Patient_Name": "", "DOB": "", "Address": "",
          "Phone_Number": "", "Medicare_ID": "", "Brace_Size": "", "Height": "",
          "Weight": "", "Pain_Level": "", "Doctor_Name": "", "Last_Visit_Date": "",
          "Previous_Treatments": "", "Score": "Numerical value (e.g., 65)", 
          "Strengths": "List only genuine strengths", 
          "Weaknesses": "Detail every error, repetition, and compliance risk found", 
          "Call_Status": "Pass (if Score >= 80) / Fail (if Score < 80)"
        }
        """
        
        response = self.model.generate_content(
            [prompt, audio_file],
            generation_config={"response_mime_type": "application/json"}
        )
        
        # تحويل النص إلى JSON
        try:
            data = json.loads(self._clean_json(response.text))
        except Exception as e:
            raise Exception(f"Failed to parse AI response: {str(e)}")
        
        # --- الحل الجذري لخطأ 'list' object has no attribute 'get' ---
        if isinstance(data, list):
            if len(data) > 0:
                data = data[0]  # استخراج أول عنصر إذا كانت النتيجة قائمة
            else:
                return {}  # إرجاع قاموس فارغ إذا كانت القائمة فارغة
        # -----------------------------------------------------------
        
        return data

# ==========================================
# 3. واجهة المستخدم (UI Handler)
# ==========================================
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
            # تحديد اللون بناءً على الحالة
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

# ==========================================
# 4. الدالة الرئيسية (Main App)
# ==========================================
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
            with st.spinner('🤖 AI Auditor is critically evaluating...'):
                # إنشاء ملف مؤقت لمعالجة الملف الصوتي
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
                    temp.write(uploaded_file.read())
                    temp_path = temp.name
                try:
                    result = analyzer.analyze_audio_final(temp_path)
                    if result:
                        st.success("✅ Analysis Complete!")
                        ui.render_results(result)
                    else:
                        st.error("AI returned empty results.")
                except Exception as e:
                    st.error(f"Analysis Error: {str(e)}")
                finally:
                    # حذف الملف المؤقت فوراً بعد الانتهاء
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
