import os
import re
from PyPDF2 import PdfReader
pdf_folder = "C:/Users/Shefali/OneDrive/Desktop/pyq oopss"
pdf_files = sorted([
os.path.join(pdf_folder, filename)
for filename in os.listdir(pdf_folder)
if filename.lower().endswith(".pdf")
], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
combined_text = ""
for pdf_file in pdf_files:
 reader = PdfReader(pdf_file)
for page in reader.pages:
 text = page.extract_text()
if text:
 combined_text += text + "\n"
question_pattern = re.compile(r'Q\s*\d+[):.\s-](.?)(?=Q\s*\d+[):.\s-]|$)', re.DOTALL)
questions = question_pattern.findall(combined_text)
cleaned_questions = [
re.sub(r'\s+', ' ', question).strip()
for question in questions if len(question.strip()) > 10
]
output_path = os.path.join(pdf_folder, "all_extracted_questions.txt")
with open(output_path, "w", encoding="utf-8") as f:
 f.write("📘 Combined Questions Extracted from PDFs\n")
 f.write("=" * 50 + "\n\n")
 for i, question in enumerate(cleaned_questions, 1):
  f.write(f"{i}. {question}\n\n")

print(f"Done! Questions saved to:\n{output_path}")
