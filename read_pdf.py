import PyPDF2

pdf_path = r"C:\Users\admin\Desktop\所有文章\Image Aesthetics Assessment With Attribute-Assisted Multimodal Memory Network(LD Li,CSVT,202312)\Image Aesthetics Assessment With Attribute-Assisted Multimodal Memory Network(LD Li,CSVT,202312).pdf"
output_path = r"C:\Users\admin\Desktop\mmlq_iaa-master\paper_content_2.txt"

with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    num_pages = len(reader.pages)

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(f"总页数: {num_pages}\n\n")
        out.write("="*80 + "\n")

        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            out.write(f"\n--- 第 {i+1} 页 ---\n\n")
            out.write(text)
            out.write("\n\n" + "="*80 + "\n")

print(f"PDF内容已提取到: {output_path}")
