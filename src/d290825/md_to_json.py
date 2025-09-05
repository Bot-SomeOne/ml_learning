import os
import json
import re

def parse_markdown(file_path):
    data = []
    section = "Introduction"
    doc_id_prefix = os.path.basename(file_path).replace(".md", "")
    idx = 1

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    buffer = []
    for line in lines:
        # Nếu là tiêu đề markdown
        if re.match(r"^#{1,6} ", line.strip()):
            # Lưu lại đoạn trước đó
            if buffer:
                content = " ".join(buffer).strip()
                if content:
                    data.append({
                        "id": f"{doc_id_prefix}-{idx}",
                        "source": os.path.basename(file_path),
                        "section": section,
                        "content": content
                    })
                    idx += 1
                buffer = []
            # Cập nhật section mới
            section = line.strip("# ").strip()
        else:
            if line.strip():
                buffer.append(line.strip())

    # Lưu đoạn cuối
    if buffer:
        content = " ".join(buffer).strip()
        if content:
            data.append({
                "id": f"{doc_id_prefix}-{idx}",
                "source": os.path.basename(file_path),
                "section": section,
                "content": content
            })

    return data

def main():
    # Load data
    folder = "data/markdown"
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".md"):
            files.append(os.path.join(folder, filename))
    # Convert to json
    all_data = []
    for file_path in files:
        all_data.extend(parse_markdown(file_path))
    output_folder = "data/json"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "all.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_data)} entries to {output_path}")
    
if __name__ == "__main__":
    main()

