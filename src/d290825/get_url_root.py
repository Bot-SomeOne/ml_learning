"""
Get root url file html
"""

import json
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import os

def extract_full_og_url(file_path):
    """Trích xuất đầy đủ URL từ thẻ og:url"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Method 1: Sử dụng BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        og_url_tag = soup.find('meta', property='og:url')
        
        if og_url_tag and og_url_tag.get('content'):
            return og_url_tag['content'].strip()
        
        # Method 2: Backup với regex
        pattern = r'<meta\s+property=["\']og:url["\']\s+content=["\']([^"\']+)["\']'
        match = re.search(pattern, content)
        if match:
            return match.group(1).strip()
            
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
    
    return None

def main():
    folder_html = "data/html"
    file_save = "data/url.json"
    # Load folder
    if not os.path.exists(folder_html):
        print(f"Folder {folder_html} does not exist.")
        return
    # Get file paths
    files = []
    for file in os.listdir(folder_html):
        if file.endswith(".html"):
            files.append(file)
    #  Get url content
    url_root = []
    for f in files:
        file_path = os.path.join(folder_html, f)
        root = extract_full_og_url(file_path)
        if root:
            url_root.append({
                "file": f,
                "url": root
            })
    # Save file
    with open(file_save, "w", encoding="utf-8") as f:
        json.dump(url_root, f, ensure_ascii=False, indent=4)
        
    print(url_root)
    
if __name__ == "__main__":
    main()