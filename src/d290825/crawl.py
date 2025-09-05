import json
import os
import random
import time
import requests
from bs4 import BeautifulSoup

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

    def getHtmlMainContent(self, filePath: str) -> None:
        """
            Get main content html
        """
        # Load data in file html
        f = open(filePath, "r", encoding="utf-8")
        htmlData = f.read()
        f.close()
        # Get main content - Get itemprop="articleBody"
        soup = BeautifulSoup(htmlData, "html.parser")
        article_body = soup.find(attrs={"itemprop": "articleBody"})
        if article_body:
            data = str(article_body)
            # Save main content to file
            main_content_path = filePath.replace(".html", ".main.html")
            with open(main_content_path, "w", encoding="utf-8") as f:
                f.write(data)

        return None

    def check_file_data_exists(self, filePath: str) -> bool:
        """
            Check if file data exists
        """
        return os.path.exists(filePath)

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
            if not self.check_file_data_exists(f"data/html/{id}.html"):
                if self.get_data(f"{id}.html", "data/html", self.url + url):
                    print(f"Downloaded {title}")
                    self.getHtmlMainContent(f"data/html/{id}.html")
                else:
                    print(f"Failed to download {title}")
            else:
                self.getHtmlMainContent(f"data/html/{id}.html")
            time.sleep(random.uniform(2, 5))

def main() -> None:
    """
        Main function
    """

    htmlData = GetHtml("https://machinelearningcoban.com/", "lesson.json")
    htmlData.run()
    
    
if __name__ == "__main__":
    main()
