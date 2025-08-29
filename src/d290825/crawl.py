import json
import os
import random
import time
import requests

"""
    Crawl data https://machinelearningcoban.com/archive/
"""

class GetHtml:
    """
        Get html data
    """
    def __init__(self, urlBase: str, srcData: str) -> None:
        self.url = urlBase
        self.srcData = srcData

    def get_data(self, fileName: str, folder: str, url: str) -> bool:
        """
            Get all data html file and save to folder
        """
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(folder, exist_ok=True)
        htmlData = requests.get(url).text
        filePath = f"{folder}/{fileName}"
        # create file and write
        with open(filePath, "w", encoding="utf-8") as f:
            f.write(htmlData)
        return True

    def run(self) -> None:
        # load data json in file
        with open(self.srcData, "r", encoding="utf-8") as f:
            jsonData = f.read()
        # print(jsonData)
        arr = json.loads(jsonData)
        for item in arr:
            id = item["id"]
            title = item["title"]
            createAt = item["date"]
            url = item["url"]
            # print(f"Id: {id}, Title: {title}, Created At: {createAt}, URL: {url}")
            if self.get_data(f"{id}.html", "data/html", self.url + url):
                print(f"Downloaded {title}")
            else:
                print(f"Failed to download {title}")
            time.sleep(random.uniform(2, 5))

def main() -> None:
    """
        Main function
    """

    # htmlData = GetHtml("https://machinelearningcoban.com/", "lesson.json")
    # htmlData.run()
    
    
    
if __name__ == "__main__":
    main()
