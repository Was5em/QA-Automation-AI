import streamlit as st
import google.generativeai as genai
import json
import os
import tempfile
import time
from typing import Dict, Any, List
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

        STRICT JSON OUTPUT SCHEMA:
        {
          "Agent_Name": "<String: Extract agent's name>",
          "Call_Date": "<String: Extract date of call>",
          "Patient_Name": "<String: Extract patient's full name>",
          "DOB": "<String: Extract Date of Birth>",
          "Address": "<String: Extract full address>",
          "Phone_Number": "<String: Extract phone number>",
          "Medicare_ID": "<String: Extract Medicare ID if mentioned>",
          
          "Medical_History": {
            "Kidney_Disease": "<YES/NO/NOT_MENTIONED>",
            "Cancer": "<YES/NO/NOT_MENTIONED>",
            "Memory_Loss": "<YES/NO/NOT_MENTIONED>",
            "Cognitive_Impairment": "<YES/NO/NOT_MENTIONED>",
            "Caregiver": "<YES/NO - Provide reason if NO>",
            "Medications": ["<String: List medication 1 (specify if OTC/Prescription)>"],
            "Arthritis_Details": {
              "Affected_Joints": ["<String: Joint 1>"],
              "Pain_Descriptor": "<String: Patient's EXACT words for pain>",
              "Pain_Triggers": ["<String: Trigger 1>"],
              "Pain_Pattern": "<String: Time of day or progression>"
            },
            "Surgical_History": [
              {
                "Procedure": "<String: Exact procedure name>",
                "Date": "<String: Exact date or timeframe>",
                "Side": "<String: Left/Right/Bilateral/NA>",
                "Status": "<Completed/Pending>",
                "Patient_Expectation": "<String: Patient's belief about this surgery>"
              }
            ]
          },
          
          "Doctor_Details": [
            {
              "Name": "<String: Exact spelling or phonetic equivalent>",
              "Name_Confidence": "<HIGH/MEDIUM/LOW>",
              "Specialty": "<String: PCP, Orthopedist, etc.>",
              "Facility": "<String: Clinic or location name>",
              "Last_Visit": "<String: Exact timeframe mentioned>",
              "Next_Appointment": "<String: Exact date or timeframe>",
              "Selection_Reason": "<String: Why did patient choose this doctor?>",
              "DME_Awareness": {
                "Cane": "<YES/NO/Unknown>",
                "Cane_Referred_By_Provider": "<YES/NO/Not specified>",
                "Walker": "<YES/NO/Unknown>",
                "Walker_Referred_By_Provider": "<YES/NO/Not specified>",
                "Brace": "<YES/NO/Unknown>",
                "Brace_Referred_By_Provider": "<YES/NO/Not specified>"
              }
            }
          ],
          
          "Objection_Handling": [
            {
              "Objection_Number": "<Integer>",
              "Category": "<String>",
              "Patient_Reasoning": "<String: Quote or paraphrase the exact hesitation>",
              "Agent_Response": "<String: Specific strategy or phrase used by agent>",
              "Resolution": "<Overcome/Partial/Not Overcome>",
              "Patient_Final_Position": "<Accepted/Still Hesitant/Refused>"
            }
          ],
          
          "Equipment_Details": {
            "Brace_Size": "<String>",
            "Waist_Size": "<String>",
            "Height": "<String>",
            "Weight": "<String>"
          },
          
          "Score": "<Integer: 0-100>",
          "Call_Status": "<Pass/Fail>",
          
          "Detailed_Analysis": {
            "Strengths": ["<String>"],
            "Weaknesses": ["<String>"],
            "Narrative": "<String: Objective microscopic summary of the interaction>"
          }
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
            pdf.cell(0, 8, f"Agent: {PDFManager._sanitize(res.get('Agent_Name'))} | Date: {PDFManager._sanitize(res.get('Call_Date'))}", ln=True)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"Score: {res.get('Score', 'N/A')}/100 | Status: {res.get('Call_Status', 'N/A')}", ln=True)
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, " Medical & Clinical History", ln=True, fill=True)
            pdf.set_font("Arial", '', 12)
            
            med = res.get('Medical_History', {})
            arth = med.get('Arthritis_Details', {})
            med_text = (f"Kidney: {med.get('Kidney_Disease')} | Cancer: {med.get('Cancer')} | "
                        f"Memory Loss: {med.get('Memory_Loss')} | Caregiver: {med.get('Caregiver')}\n"
                        f"Medications: {', '.join(med.get('Medications', []))}\n"
                        f"Arthritis: Joints({', '.join(arth.get('Affected_Joints', []))}), "
                        f"Descriptor({arth.get('Pain_Descriptor')}), Triggers({', '.join(arth.get('Pain_Triggers', []))})")
            pdf.multi_cell(0, 8, PDFManager._sanitize(med_text))
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Surgical History:", ln=True)
            pdf.set_font("Arial", '', 12)
            for surg in res.get('Medical_History', {}).get('Surgical_History', []):
                pdf.multi_cell(0, 8, PDFManager._sanitize(f"- {surg.get('Procedure')} ({surg.get('Date')}) Side: {surg.get('Side')} | Status: {surg.get('Status')}"))
            
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, " Provider Details", ln=True, fill=True)
            pdf.set_font("Arial", '', 12)
            for doc in res.get('Doctor_Details', []):
                dme = doc.get('DME_Awareness', {})
                doc_text = (f"Dr. {doc.get('Name')} ({doc.get('Specialty')}) | Visit: {doc.get('Last_Visit')}\n"
                            f"Selection Reason: {doc.get('Selection_Reason')}\n"
                            f"DME Awareness: Cane({dme.get('Cane')}), Walker({dme.get('Walker')}), Brace({dme.get('Brace')})")
                pdf.multi_cell(0, 8, PDFManager._sanitize(doc_text))
                pdf.ln(2)

            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, " Objection Handling", ln=True, fill=True)
            pdf.set_font("Arial", '', 12)
            for obj in res.get('Objection_Handling', []):
                obj_text = (f"#{obj.get('Objection_Number')} [{obj.get('Category')}]: {obj.get('Patient_Reasoning')}\n"
                            f"Agent Response: {obj.get('Agent_Response')} -> Result: {obj.get('Resolution')}")
                pdf.multi_cell(0, 8, PDFManager._sanitize(obj_text))
                pdf.ln(2)

            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, " Final Auditor's Narrative", ln=True, fill=True)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, PDFManager._sanitize(res.get('Detailed_Analysis', {}).get('Narrative', 'N/A')))
            
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Strengths:", ln=True)
            pdf.set_font("Arial", '', 12)
            for s in res.get('Detailed_Analysis', {}).get('Strengths', []):
                pdf.cell(0, 8, f"- {PDFManager._sanitize(s)}", ln=True)
            
            pdf.ln(2)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Weaknesses:", ln=True)
            pdf.set_font("Arial", '', 12)
            for w in res.get('Detailed_Analysis', {}).get('Weaknesses', []):
                pdf.cell(0, 8, f"- {PDFManager._sanitize(w)}", ln=True)
            
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
            st.markdown(f"""<div class="custom-card"><div class="card-title">👤 Agent & Call</div>
                <div style="font-size: 1.1rem;"><span class="data-label">Agent:</span> <span class="data-value">{res.get('Agent_Name', 'N/A')}</span><br>
                <span class="data-label">Call Date:</span> <span class="data-value">{res.get('Call_Date', 'N/A')}</span></div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="card-title">📝 Senior Auditor\'s Narrative</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="narrative-box">{res.get("Detailed_Analysis", {}).get("Narrative", "N/A")}</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-title">🏥 Clinical Intelligence</div>', unsafe_allow_html=True)
        med = res.get('Medical_History', {})
        st.markdown(f"""<div class="custom-card">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div><span class="data-label">Memory/Cognitive:</span> <span class="data-value">{med.get('Memory_Loss')} / {med.get('Cognitive_Impairment')}</span></div>
                <div><span class="data-label">Caregiver:</span> <span class="data-value">{med.get('Caregiver')}</span></div>
                <div><span class="data-label">Kidney/Cancer:</span> <span class="data-value">{med.get('Kidney_Disease')} / {med.get('Cancer')}</span></div>
                <div><span class="data-label">Medications:</span> <span class="data-value">{', '.join(med.get('Medications', []))}</span></div>
            </div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="card-title">🔄 Objection Handling Map</div>', unsafe_allow_html=True)
        obj_html = "".join([f'<div class="custom-card"><b>#{o.get("Objection_Number")} {o.get("Category")}:</b><br>{o.get("Patient_Reasoning")} <br><b>Resolution:</b> {o.get("Agent_Response")} &rarr; {o.get("Resolution")}</div>' for o in res.get('Objection_Handling', [])])
        st.markdown(obj_html, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🌟 Strengths", "⚠️ Weaknesses"])
        with tab1: 
            for s in res.get('Detailed_Analysis', {}).get('Strengths', []): st.success(s)
        with tab2: 
            for w in res.get('Detailed_Analysis', {}).get('Weaknesses', []): st.error(w)

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
