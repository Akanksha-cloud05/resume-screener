# test_short.py
import fitz

doc = fitz.open()
page = doc.new_page()
page.insert_textbox(fitz.Rect(50, 50, 550, 100), "Python developer. Short resume.", fontsize=12)
doc.save("short_resume.pdf")
doc.close()

print("Created short_resume.pdf")