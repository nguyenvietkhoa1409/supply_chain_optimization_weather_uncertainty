import docx
doc = docx.Document(r'D:\Food chain optimization\Materials\fixing plan.docx')
for p in doc.paragraphs:
    if p.text.strip():
        print(p.text)
