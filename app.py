import streamlit as st
import google.generativeai as genai
import json
import os
import tempfile
import time
import io
from typing import Dict, Any, Optional
from fpdf import FPDF

class QAConfig:
    API_KEY = st.secrets.get("GOOGLE_API_KEY", "AIzaSyDjOP3Ps9lsLAeEp5bgexGMAn7AJqn04Ek")
    MODEL_NAME = 'models/gemini-flash-latest'
    PAGE_TITLE = "Medical Call QA Dashboard"
    PAGE_ICON = "🩺"

class QAAnalyzer:
    def __init__(self):
        genai.configure(api_key=QAConfig.API_KEY)
        self.model = genai.GenerativeModel(QAConfig.MODEL_NAME)

    def _clean_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def analyze_audio_final(self, file_path: str) -> Dict[str, Any]:
        audio_file = genai.upload_file(path=file_path)
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = genai.get_file(audio_file.name)
        
        prompt = """
        Act as an expert Senior Medical QA Auditor. Perform a microscopic analysis of the call. 
        Capture every clinical and behavioral nuance without summarizing.

        ### EXTRACTION REQUIREMENTS:
        1. MEDICAL VETTING: 
           - Check for Kidney disease, Cancer, Memory loss, or Cognitive impairment.
           - Identify any Caregiver involvement.
           - List all medications (including OTC like Tylenol).
           - Detail Arthritis: Which joints? Pain description (e.g., achy)? Triggers (e.g., stairs, walking)?

        2. PROVIDER LOGIC:
           - Doctor names, visit timelines (e.g., 3 months ago), and selection logic (who chose the doctor?).
           - Determine if the doctor was aware of current aids (e.g., cane) and if the aid was referred by the doctor.

        3. OBJECTION HANDLING:
           - Map every patient concern to the agent's resolution.
           - Specifically capture: Refusals due to previous surgeries (e.g., knee replacement), suspicion, or need to consult a doctor first.
           - Analyze how the agent handled "forgetfulness" or suspicion.

        OUTPUT FORMAT (Strict JSON):
        {
          "Agent_Name": "", "Patient_Name": "", "DOB": "", "Address": "",
          "Phone_Number": "", "Medicare_ID": "", "Brace_Size": "", "Waist_Size": "",
          "Height": "", "Weight": "", "Pain_Level": "", 
          "Pain_Details": "Description and triggers",
          "Medical_History": "Kidney, Cancer, Memory, Caregiver, Medications",
          "Doctor_Details": "Names, timelines, and referral logic",
          "Objection_Handling": "Detailed map of Concern -> Resolution",
          "Score": "Numerical value", 
          "Detailed_Analysis": "Professional narrative connecting behavior to patient reactions",
          "Strengths": "Key positives", 
          "Weaknesses": "Key negatives", 
          "Call_Status": "Pass/Fail"
        }
        """
        
        response = self.model.generate_content(
            [prompt, audio_file],
            generation_config={"response_mime_type": "application/json"}
        )
        
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
        replacements = {
            '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'", 
            '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u00a0': ' '
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
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
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, " General Overview", ln=True, fill=True)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 8, f"Agent: {PDFManager._sanitize(res.get('Agent_Name'))}", ln=True)
            pdf.cell(0, 8, f"Date: {time.strftime('%Y-%m-%d %H:%M')}", ln=True)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"Score: {res.get('Score', 'N/A')}/100 | Status: {res.get('Call_Status', 'N/A')}", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, " Patient Clinical Data", ln=True, fill=True)
            pdf.set_font("Arial", '', 12)
            
            clinical_data = [
                ("Patient", res.get('Patient_Name')),
                ("DOB", res.get('DOB')),
                ("Medical History", res.get('Medical_History')),
                ("Pain Details", res.get('Pain_Details')),
                ("Doctor Info", res.get('Doctor_Details')),
                ("Sizes (Brace/Waist)", f"{res.get('Brace_Size')} / {res.get('Waist_Size')}"),
                ("Height/Weight", f"{res.get('Height')} / {res.get('Weight')}"),
            ]
            for label, val in clinical_data:
                pdf.multi_cell(0, 8, f"{label}: {PDFManager._sanitize(val)}")
            
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, " Objection Handling & Resolution", ln=True, fill=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, PDFManager._sanitize(res.get('Objection_Handling')))
            
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, " Senior Auditor's Narrative", ln=True, fill=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, PDFManager._sanitize(res.get('Detailed_Analysis')))
            
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Strengths:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, PDFManager._sanitize(res.get('Strengths')))
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Weaknesses:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, PDFManager._sanitize(res.get('Weaknesses')))
            
            output = pdf.output(dest='S')
            return output.encode('latin-1') if isinstance(output, str) else output
        except Exception as e:
            st.error(f"PDF Generation Error: {str(e)}")
            return b""

class UIHandler:
    @staticmethod
    def apply_styles():
        st.markdown("""
            <style>
            .stApp { background-color: #f8f9fa; }
            .main-header {
                background: linear-gradient(90deg, #0f172a 0%, #2563eb 100%);
                color: white; padding: 2rem; border-radius: 15px;
                text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .main-header h1 { color: white !important; font-size: 2.5rem !important; margin-bottom: 0; }
            .custom-card {
                background-color: white; padding: 20px; border-radius: 15px;
                border-left: 5px solid #2563eb; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                margin-bottom: 20px;
            }
            .narrative-box {
                background-color: #ffffff; padding: 25px; border-radius: 15px;
                border: 1px solid #d1d5db; font-family: 'Georgia', serif;
                line-height: 1.7; color: #334155; font-size: 1.1rem; margin-bottom: 20px;
            }
            .card-title { color: #1e3a8a; font-size: 1.3rem; font-weight: bold; margin-bottom: 15px; }
            .data-label { font-weight: 600; color: #64748b; width: 160px; display: inline-block; }
            .data-value { color: #1e293b; }
            </style>
            """, unsafe_allow_html=True)

    @staticmethod
    def render_header():
        st.markdown('<div class="main-header"><h1>🩺 Medical Call QA Dashboard</h1><p>Powered by Outsourcing Skill</p></div>', unsafe_allow_html=True)

    @staticmethod
    def render_results(res: Dict[str, Any]):
        col1, col2 = st.columns([1, 2])
        with col1:
            color = "green" if res.get("Call_Status") == "Pass" else "red"
            st.markdown(f"""<div class="custom-card"><div class="card-title">📊 Score</div>
                <div style="text-align:center;"><h2 style="font-size: 3rem; color: #1e3a8a; margin: 0;">{res.get('Score', 'N/A')}/100</h2>
                <p style="color: {color}; font-weight: bold;">Status: {res.get('Call_Status', 'N/A')}</p></div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="custom-card"><div class="card-title">👤 Agent</div>
                <div style="font-size: 1.1rem;"><span class="data-label">Name:</span> <span class="data-value">{res.get('Agent_Name', 'N/A')}</span><br>
                <span class="data-label">Date:</span> <span class="data-value">{time.strftime('%Y-%m-%d %H:%M')}</span></div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="card-title">📝 Senior Auditor\'s Narrative</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="narrative-box">{res.get("Detailed_Analysis", "N/A")}</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-title">🏥 Clinical Intelligence</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="custom-card">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div><span class="data-label">Medical History:</span> <span class="data-value">{res.get('Medical_History', 'N/A')}</span></div>
                <div><span class="data-label">Pain Details:</span> <span class="data-value">{res.get('Pain_Details', 'N/A')}</span></div>
                <div><span class="data-label">Doctor Info:</span> <span class="data-value">{res.get('Doctor_Details', 'N/A')}</span></div>
                <div><span class="data-label">Waist/Brace:</span> <span class="data-value">{res.get('Waist_Size', 'N/A')} / {res.get('Brace_Size', 'N/A')}</span></div>
            </div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="card-title">🔄 Objection Handling</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="custom-card">{res.get("Objection_Handling", "N/A")}</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🌟 Strengths", "⚠️ Weaknesses"])
        with tab1: st.success(res.get("Strengths", "N/A"))
        with tab2: st.error(res.get("Weaknesses", "N/A"))

        st.markdown('<div class="card-title" style="margin-top: 2rem;">📥 Export Full Report</div>', unsafe_allow_html=True)
        pdf_data = PDFManager.create_full_pdf(res)
        st.download_button(label="📄 Download Complete Analysis PDF", data=pdf_data, 
                           file_name=f"Medical_QA_{res.get('Agent_Name', 'Agent')}.pdf", 
                           mime="application/pdf", use_container_width=True)

def main():
    st.set_page_config(page_title=QAConfig.PAGE_TITLE, page_icon=QAConfig.PAGE_ICON, layout="wide")
    ui = UIHandler()
    ui.apply_styles()
    ui.render_header()
    analyzer = QAAnalyzer()
    
    st.sidebar.header("📂 Upload Call Record")
    uploaded_file = st.sidebar.file_uploader("Upload MP3/WAV", type=["mp3", "wav"])
    
    if uploaded_file:
        st.sidebar.audio(uploaded_file, format='audio/mp3')
        if st.sidebar.button("🚀 Analyze Call Now"):
            with st.spinner('🤖 Auditor is drafting the report...'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp:
                    temp.write(uploaded_file.read())
                    temp_path = temp.name
                try:
                    result = analyzer.analyze_audio_final(temp_path)
                    if result:
                        st.success("✅ Analysis Complete!")
                        ui.render_results(result)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    if os.path.exists(temp_path): os.remove(temp_path)
    else:
        st.info("👈 Please upload an audio file to begin.")

if __name__ == "__main__":
    main()
