import os
import markdown2
from fpdf import FPDF

def md_to_pdf(md_path, pdf_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("helvetica", size=12)
    
    # Simple clean up for latin-1
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf.multi_cell(0, 10, safe_text)
    pdf.output(pdf_path)

def main():
    # Directories to search
    search_dirs = [
        os.getcwd(),
        os.path.join(os.getcwd(), "testing", "DETECTIFY"),
        os.path.join(os.getcwd(), "testing_integration", "DETECTIFY"),
        r"C:\Users\theof\.gemini\antigravity\brain\cb9bc851-1063-4234-ab80-4e3544c1929a"
    ]
    
    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            print(f"Directory not found: {s_dir}")
            continue
        print(f"Searching in {s_dir}...")
        for filename in os.listdir(s_dir):
            if filename.endswith(".md"):
                md_path = os.path.join(s_dir, filename)
                pdf_path = os.path.join(s_dir, filename.replace(".md", ".pdf"))
                print(f"Converting {filename} to PDF...")
                try:
                    md_to_pdf(md_path, pdf_path)
                    print(f"Successfully converted {filename}")
                except Exception as e:
                    print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    main()
