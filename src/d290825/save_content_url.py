"""
Get root url file html
"""

import json
import os

def main():
    json_file = "data/url.json"
    json_file_data = "data/json/all.json"
    file_save = "data/data.json"
    #  Get data in file and change 
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data_url = json.load(f)
        with open(json_file_data, 'r', encoding='utf-8') as f:
            data  = json.load(f)
        # Chuyển đổi dữ liệu
        transformed_data = []
        for d in data:
            source = d.get('source', '') # 22.main.md
            for u in data_url:
                file_idx = u.get('file', '') # 22.html
                if source.split('.')[0] == file_idx.split('.')[0]:
                    d['url'] = u.get('url', '')
                    transformed_data.append(d)
                    continue
                            
        # Lưu dữ liệu đã chuyển đổi vào file mới
        with open(file_save, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=4)
        
        print(f"Dữ liệu đã được lưu vào {file_save}")
    except Exception as e:
        print(f"Lỗi khi xử lý file {json_file}: {e}")

if __name__ == "__main__":
    main()