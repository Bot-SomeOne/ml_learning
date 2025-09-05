import markdownify
import os


class HTMLToMarkdown:
    """
        File covert html to markdown
    """
    def __init__(self, html_content: str):
        self.html_content = html_content

    def convert(self) -> str:
        return markdownify.markdownify(self.html_content, heading_style="ATX")
    
def main():
    """
        Main function
    """
    folder_html = "data/main_html/"
    folder_save_md = "data/markdown/"
    #  Check folder path
    if not os.path.exists(folder_html):
        print(f"Folder {folder_html} does not exist.")
        return
    files = []
    for file in os.listdir(folder_html):
        if file.endswith(".html"):
            files.append(file)
    print(f"Found {len(files)} HTML files in {folder_html}")
    for file in files:
        file_path = os.path.join(folder_html, file)
        # read file and convert html to markdown
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            converter = HTMLToMarkdown(content)
            markdown_content = converter.convert()
            md_filename = os.path.splitext(file)[0] + ".md"
            md_file_path = os.path.join(folder_save_md, md_filename)
            os.makedirs(folder_save_md, exist_ok=True)
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(markdown_content)
            print(f"Converted {file} to {md_filename}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
    
    print("Done !")

if __name__ == "__main__":
    main()