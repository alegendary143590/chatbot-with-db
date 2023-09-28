from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.parse import urlparse
from bs4 import BeautifulSoup


def extract_text_content(url):
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--incognito')
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)

    width = 1080
    height = 1920
    driver.set_window_size(height, width)

    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace(".", "_")
    path = parsed_url.path.strip("/").replace("/", "_")
    file_name = domain + "_" + path + ".html"

    driver.get(url)

    title = driver.title
    page_source = driver.page_source

    soup = BeautifulSoup(page_source, 'html.parser')

    header = soup.find('header')
    if header:
        header.extract()

    nav = soup.find('nav')
    if nav:
        nav.extract()

    footer = soup.find('footer')
    if footer:
        footer.extract()

    for script in soup(["script", "style"]):
        script.extract()

    text_content = soup.get_text().strip()
    text_content = ' '.join(text_content.split())

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <title>{title}</title>
    </head>
    <body>
    <p>{text_content}</p>
    </body>
    </html>
    """

    file_ = open(file_name, 'w', encoding='utf-8')
    file_.write(html_content)
    file_.close()

    driver.close()

    return file_name


url = input("URL: ")
output_file = extract_text_content(url)
print("Extraction completed. HTML file saved as:", output_file)
