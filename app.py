import streamlit as st
import google.generativeai as genai
import json
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, Any, List
from fpdf import FPDF

class QAConfig:
    API_KEY = st.secrets.get("GOOGLE_API_KEY", "AIzaSyDjOP3Ps9lsLAeEp5bgexGMAn7AJqn04Ek")
    ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123") 
    PRICE_INPUT_1M = 0.075  
    PRICE_OUTPUT_1M = 0.30  
    USAGE_FILE = "usage_log.json"
    MODEL_NAME = 'gemini-1.5-flash'
    PAGE_TITLE = "Medical Call QA Dashboard"
    PAGE_ICON = "🩺"

class UsageTracker:
    @staticmethod
    def log_usage(prompt_tokens: int, response_tokens: int):
        today = datetime.now().strftime("%Y-%m-%d")
        data = UsageTracker.load_logs()
        if today not in data:
            data[today] = {"input": 0, "output": 0, "cost": 0.0}
        data[today]["input"] += prompt_tokens
        data[today]["output"] += response_tokens
        cost = (prompt_tokens * (QAConfig.PRICE_INPUT_1M / 1_000_000)) + \
               (response_tokens * (QAConfig.PRICE_OUTPUT_1M / 1_000_000))
        data[today]["cost"] += cost
        with open(QAConfig.USAGE_FILE, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load_logs():
        if os.path.exists(QAConfig.USAGE_FILE):
            with open(QAConfig.USAGE_FILE, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def get_today_stats():
        today = datetime.now().strftime("%Y-%m-%d")
        logs = UsageTracker.load_logs()
        return logs.get(today, {"input": 0, "output": 0, "cost": 0.0})

class QAAnalyzer:
    def __init__(self):
        genai.configure(api_key=QAConfig.API_KEY)
        self.model = genai.GenerativeModel(QAConfig.MODEL_NAME)

    def _clean_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.endswith("```"): text = text[:-3]
        return text.strip()

    def analyze_audio_final(self, file_path: str) -> Dict[str, Any]:
        audio_file = genai.upload_file(path=file_path)
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = genai.get_file(audio_file.name)
        
        # --- البرومبت الجديد المعتمد كلياً على السكريبت المرفق ---
        prompt = """
        Act as a strict Senior Medical QA Auditor. Your sole reference for this audit is the OFFICIAL CALL SCRIPT.
        
        OFFICIAL SCRIPT GUIDELINES:
        1. Introduction: Must use one of the 5 options (Identify patient, introduce as medical alert specialist, check for existing system).
        2. Scenario Path:
           - If NO system: Must ask about HBP/HCL/Diabetes, chronic conditions, balance/dizziness, medications, and living alone.
           - If YES system: Must ask about device type (necklace), monthly cost, and experience.
        3. Product Description: Must mention 24/7 protection, water-resistance, and automatic fall detection.
        4. Smartwatch Add-on: If watch is chosen, must mention BP, heart rate, steps, and medication reminders.
        5. Pricing: Must state NO UPFRONT COST. Necklace = $40/mo, Smartwatch = $45/mo.
        6. Info Gathering: Must collect Phone Type, DOB, Shipping Address, Payment Method, Emergency Contact, and Old Company (if applicable).
        7. Closing/Transfer: Must perform the Disclosure and receive a clear 'YES'.
        8. Rebuttals: Must use the specific rebuttals provided for objections like 'can't afford' or 'not interested'.

        AUDIT TASK:
        - Check if the agent followed the sequence and wording of the script.
        - Identify exactly which script steps were missed.
        - Extract all captured patient data.

        STRICT JSON OUTPUT SCHEMA:
        {
          "Agent_Name": "<String>",
          "Call_Date": "<String>",
          "Patient_Name": "<String>",
          "DOB": "<String>",
          "Address": "<String>",
          "Phone_Number": "<String>",
          "Medicare_ID": "<String>",
          "Script_Adherence": {
            "Intro_Correct": "<YES/NO>",
            "Correct_Scenario_Used": "<Scenario A / Scenario B / NO>",
            "Health_Questions_Asked": "<YES/NO>",
            "Product_Description_Complete": "<YES/NO>",
            "Pricing_Quoted_Correctly": "<YES/NO>",
            "Info_Gathering_Complete": "<YES/NO>",
            "Disclosure_Confirmed_YES": "<YES/NO>",
            "Missing_Steps": ["<List specific missing parts of the script>"]
          },
          "Medical_History": {
            "Kidney_Disease": "<YES/NO/NOT_MENTIONED>",
            "Cancer": "<YES/NO/NOT_MENTIONED>",
            "Memory_Loss": "<YES/NO/NOT_MENTIONED>",
            "Cognitive_Impairment": "<YES/NO/NOT_MENTIONED>",
            "Caregiver": "<YES/NO>",
            "Medications": ["<String>"],
            "Arthritis_Details": { "Affected_Joints": [], "Pain_Descriptor": "", "Pain_Triggers": [], "Pain_Pattern": "" },
            "Surgical_History": [ { "Procedure": "", "Date": "", "Side": "", "Status": "", "Patient_Expectation": "" } ]
          },
          "Doctor_Details": [ { "Name": "", "Specialty": "", "Last_Visit": "", "Selection_Reason": "", "DME_Awareness": { "Cane": "YES/NO", "Walker": "YES/NO", "Brace": "YES/NO" } } ],
          "Objection_Handling": [ { "Objection_Number": 0, "Category": "", "Patient_Reasoning": "", "Agent_Response": "", "Resolution": "" } ],
          "Equipment_Details": { "Brace_Size": "", "Waist_Size": "", "Height": "", "Weight": "" },
          "Score": <Integer>,
          "Call_Status": "<Pass/Fail>",
          "Detailed_Analysis": {
            "Strengths": ["<String>"],
            "Weaknesses": ["<String>"],
            "Narrative": "<Detailed audit narrative focusing on script adherence>"
          }
        }
        """
        response = self.model.generate_content([prompt, audio_file], generation_config={"response_mime_type": "application/json"})
        usage = response.usage_metadata
        UsageTracker.log_usage(usage.prompt_token_count, usage.candidates_token_count)
        try:
            data = json.loads(self._clean_json(response.text))
            return data[0] if isinstance(data, list) else data
        except Exception as e:
            raise Exception(f"AI Parsing Error: {str(e)}")

class PDFManager:
    @staticmethod
    def _sanitize(text: Any) -> str:
        if text is None: return "N/A"
        text = str(text)
        replacements = {'\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u00a0': ' '}
        for bad, good in replacements.items(): text = text.replace(bad, good)
        return text.encode('latin-1', 'replace').decode('latin-1')

    @staticmethod
    def create_full_pdf(res: Dict[str, Any]) -> bytes:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 20)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 15, "Medical Call QA Full Report", ln=True, align='C')
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(230, 235, 245)
            pdf.cell(0, 10, " General Overview", ln=True, fill=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, f"Agent Name: {PDFManager._sanitize(res.get('Agent_Name'))}", ln=True)
            pdf.cell(0, 8, f"Overall Score: {res.get('Score', 'N/A')}/100", ln=True)
            pdf.cell(0, 8, f"Call Status: {res.get('Call_Status', 'N/A')}", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(230, 235, 245)
            pdf.cell(0, 10, " Script Adherence", ln=True, fill=True)
            pdf.set_font("Arial", '', 11)
            ad = res.get('Script_Adherence', {})
            pdf.cell(0, 8, f"Intro Correct: {ad.get('Intro_Correct')}", ln=True)
            pdf.cell(0, 8, f"Scenario Used: {ad.get('Correct_Scenario_Used')}", ln=True)
            pdf.cell(0, 8, f"Health Questions Asked: {ad.get('Health_Questions_Asked')}", ln=True)
            pdf.cell(0, 8, f"Product Description Complete: {ad.get('Product_Description_Complete')}", ln=True)
            pdf.cell(0, 8, f"Pricing Correct: {ad.get('Pricing_Corrected')}", ln=True)
            pdf.cell(0, 8, f"Disclosure Confirmed: {ad.get('Disclosure_Confirmed_YES')}", ln=True)
            pdf.multi_cell(0, 8, f"Missing Steps: {', '.join(ad.get('Missing_Steps', []))}")
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(230, 235, 245)
            pdf.cell(0, 10, " Patient & Equipment Details", ln=True, fill=True)
            pdf.set_font("Arial", '', 11)
            pdf.cell(0, 8, f"Patient Name: {PDFManager._sanitize(res.get('Patient_Name'))}", ln=True)
            pdf.multi_cell(0, 8, f"Address: {PDFManager._sanitize(res.get('Address'))}")
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(230, 235, 245)
            pdf.cell(0, 10, " Senior Auditor's Narrative", ln=True, fill=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, PDFManager._sanitize(res.get('Detailed_Analysis', {}).get('Narrative', 'N/A')))
            
            output = pdf.output(dest='S')
            return output.encode('latin-1') if isinstance(output, str) else output
        except Exception as e:
            st.error(f"PDF Error: {str(e)}")
            return b""

class UIHandler:
    @staticmethod
    def apply_styles():
        st.markdown("""
            <style>
            .stApp { background-color: #f8f9fa; }
            .main-header { background: linear-gradient(90deg, #0f172a 0%, #2563eb 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            .main-header h1 { color: white !important; font-size: 2.5rem !important; margin-bottom: 0; }
            .custom-card { background-color: white; padding: 20px; border-radius: 15px; border-left: 5px solid #2563eb; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
            .narrative-box { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #d1d5db; font-family: 'Georgia', serif; line-height: 1.7; color: #334155; font-size: 1.1rem; margin-bottom: 20px; }
            .card-title { color: #1e3a8a; font-size: 1.3rem; font-weight: bold; margin-bottom: 15px; }
            .data-label { font-weight: 600; color: #64748b; width: 160px; display: inline-block; }
            .equipment-row { display: flex; justify-content: space-around; text-align: center; background: #eff6ff; padding: 15px; border-radius: 10px; border: 1px dashed #2563eb; }
            .eq-item { display: flex; flex-direction: column; }
            .eq-val { font-weight: bold; color: #1e3a8a; font-size: 1.1rem; }
            .stat-card { background: #ffffff; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e2e8f0; }
            </style>
            """, unsafe_allow_html=True)

    @staticmethod
    def render_header():
        st.markdown('<div class="main-header"><h1>🩺 Medical Call QA Dashboard</h1><p>Powered by Outsourcing Skill</p></div>', unsafe_allow_html=True)

    @staticmethod
    def render_usage_dashboard():
        st.markdown('<div class="card-title">💰 Daily API Consumption (Estimated)</div>', unsafe_allow_html=True)
        stats = UsageTracker.get_today_stats()
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(f'<div class="stat-card"><b>Input Tokens</b><br><span style="font-size:1.5rem; color:#2563eb;">{stats["input"]:,}</span></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="stat-card"><b>Output Tokens</b><br><span style="font-size:1.5rem; color:#2563eb;">{stats["output"]:,}</span></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="stat-card"><b>Est. Cost Today</b><br><span style="font-size:1.5rem; color:#dc2626; font-weight:bold;">${stats["cost"]:.4f}</span></div>', unsafe_allow_html=True)

    @staticmethod
    def render_verification_step(res: Dict[str, Any]):
        st.markdown('<div class="card-title">🔍 Verify & Correct Extracted Names</div>', unsafe_allow_html=True)
        with st.form("verification_form"):
            col1, col2, col3 = st.columns(3)
            with col1: agent_name = st.text_input("Agent Name", res.get('Agent_Name', ''))
            with col2: patient_name = st.text_input("Patient Name", res.get('Patient_Name', ''))
            with col3: call_date = st.text_input("Call Date", res.get('Call_Date', ''))
            if st.form_submit_button("✅ Confirm & Generate Final Report"):
                res['Agent_Name'], res['Patient_Name'], res['Call_Date'] = agent_name, patient_name, call_date
                st.session_state.verified = True
                st.rerun()

    @staticmethod
    def render_results(res: Dict[str, Any]):
        col1, col2 = st.columns([1, 2])
        with col1:
            color = "green" if res.get("Call_Status") == "Pass" else "red"
            st.markdown(f"""<div class="custom-card"><div class="card-title">📊 Score</div><div style="text-align:center;"><h2 style="font-size: 3rem; color: #1e3a8a; margin: 0;">{res.get('Score', 'N/A')}/100</h2><p style="color: {color}; font-weight: bold;">Status: {res.get('Call_Status', 'N/A')}</p></div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="custom-card"><div class="card-title">👤 Call Info</div><div style="font-size: 1.1rem;"><span class="data-label">Agent:</span> {res.get('Agent_Name', 'N/A')}<br><span class="data-label">Date:</span> {res.get('Call_Date', 'N/A')}</div></div>""", unsafe_allow_html=True)
        
        st.markdown('<div class="card-title">📜 Script Adherence</div>', unsafe_allow_html=True)
        ad = res.get('Script_Adherence', {})
        st.markdown(f"""
            <div class="custom-card">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div><span class="data-label">Intro Correct:</span> {ad.get('Intro_Correct')}</div>
                    <div><span class="data-label">Scenario Used:</span> {ad.get('Correct_Scenario_Used')}</div>
                    <div><span class="data-label">Health Questions Asked:</span> {ad.get('Health_Questions_Asked')}</div>
                    <div><span class="data-label">Product Desc Complete:</span> {ad.get('Product_Description_Complete')}</div>
                    <div><span class="data-label">Pricing Correct:</span> {ad.get('Pricing_Corrected')}</div>
                    <div><span class="data-label">Disclosure Confirmed:</span> {ad.get('Disclosure_Confirmed_YES')}</div>
                </div>
                <div style="margin-top:10px;"><b>Missing/Failed Steps:</b> {', '.join(ad.get('Missing_Steps', []))}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-title">👤 Patient & Equipment Details</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="custom-card"><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px;"><div><span class="data-label">Patient Name:</span> {res.get('Patient_Name', 'N/A')}</div><div><span class="data-label">DOB:</span> {res.get('DOB', 'N/A')}</div><div><span class="data-label">Phone:</span> {res.get('Phone_Number', 'N/A')}</div><div><span class="data-label">Medicare ID:</span> {res.get('Medicare_ID', 'N/A')}</div><div style="grid-column: span 2;"><span class="data-label">Address:</span> {res.get('Address', 'N/A')}</div></div><div style="border-top: 1px dashed #ccc; margin: 15px 0;"></div><div class="equipment-row"><div class="eq-item"><span class="data-label" style="width:auto;">Height</span><span class="eq-val">{res.get('Equipment_Details', {}).get('Height', 'N/A')}</span></div><div class="eq-item"><span class="data-label" style="width:auto;">Weight</span><span class="eq-val">{res.get('Equipment_Details', {}).get('Weight', 'N/A')}</span></div><div class="eq-item"><span class="data-label" style="width:auto;">Waist Size</span><span class="eq-val">{res.get('Equipment_Details', {}).get('Waist_Size', 'N/A')}</span></div><div class="eq-item"><span class="data-label" style="width:auto;">Brace Size</span><span class="eq-val">{res.get('Equipment_Details', {}).get('Brace_Size', 'N/A')}</span></div></div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 Senior Auditor\'s Narrative</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="narrative-box">{res.get("Detailed_Analysis", {}).get("Narrative", "N/A")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-top: 2rem;">📥 Export Full Report</div>', unsafe_allow_html=True)
        pdf_data = PDFManager.create_full_pdf(res)
        st.download_button(label="📄 Download Complete Analysis PDF", data=pdf_data, file_name=f"Medical_QA_{res.get('Agent_Name', 'Agent')}.pdf", mime="application/pdf", use_container_width=True)

def main():
    st.set_page_config(page_title=QAConfig.PAGE_TITLE, page_icon=QAConfig.PAGE_ICON, layout="wide")
    if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None
    if 'verified' not in st.session_state: st.session_state.verified = False
    if 'admin_mode' not in st.session_state: st.session_state.admin_mode = False
    ui = UIHandler()
    ui.apply_styles()
    ui.render_header()
    st.sidebar.header("📂 Upload Call Record")
    uploaded_file = st.sidebar.file_uploader("Upload MP3/WAV", type=["mp3", "wav"])
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔐 Admin Access"):
        if not st.session_state.admin_mode:
            pwd = st.text_input("Enter Admin Password", type="password")
            if st.button("Login as Admin"):
                if pwd == QAConfig.ADMIN_PASSWORD:
                    st.session_state.admin_mode = True
                    st.rerun()
                else:
                    st.error("Wrong password!")
        else:
            st.success("Admin Mode Active")
            if st.button("Logout Admin"):
                st.session_state.admin_mode = False
                st.rerun()
    if st.session_state.admin_mode:
        ui.render_usage_dashboard()
        st.markdown("---")
    analyzer = QAAnalyzer()
    if uploaded_file:
        st.sidebar.audio(uploaded_file, format='audio/mp3')
        if st.sidebar.button("🚀 Analyze Call Now"):
            st.session_state.verified = False
            with st.spinner('🤖 Auditor is drafting the report...'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
                    temp.write(uploaded_file.read())
                    temp_path = temp.name
                try:
                    st.session_state.analysis_result = analyzer.analyze_audio_final(temp_path)
                    st.success("✅ Analysis Complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    if os.path.exists(temp_path): os.remove(temp_path)
    if st.session_state.analysis_result:
        if not st.session_state.verified:
            ui.render_verification_step(st.session_state.analysis_result)
        else:
            ui.render_results(st.session_state.analysis_result)
    else:
        st.info("👈 Please upload an audio file to begin.")

if __name__ == "__main__":
    main()
