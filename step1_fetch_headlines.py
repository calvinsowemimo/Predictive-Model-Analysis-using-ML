import requests
from bs4 import BeautifulSoup
import csv

# URLs to scrape
websites = [
    'https://www.theguardian.com/business/live/2018/dec/31/markets-2018-worst-year-ftse-100-china-business-live',
    'https://www.theguardian.com/business/live/2019/dec/31/global-markets-rally-shares-ftse-100-pound-oil-markets-business-live',
    'https://www.theguardian.com/business/2020/mar/31/ftse-100-posts-largest-quarterly-fall-since-black-monday-aftermath',
    'https://www.theguardian.com/business/2021/dec/31/ftse-100-bounces-back-despite-covid-to-finish-143-up-in-2021',
    'https://www.theguardian.com/business/2022/dec/30/ftse-100-2022-up-share-index-pound-dollar'
]

# Path to save the CSV file
csv_file_path = r'path\news_data.csv'  # Note the raw string notation

# Headers of your CSV file defined
csv_headers = ['Source', 'Publication Date', 'Headline']

# Function to scrape a single website
def scrape_website(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Updated selectors based on The Guardian's article structure
    headline = soup.select_one('h1')
    date = soup.select_one('time')
    
    scraped_data = []
    
    if headline and date:  
        headline_text = headline.get_text(strip=True)
        publication_date = date.get_text(strip=True)
        scraped_data.append((url, publication_date, headline_text))
    
    return scraped_data

# Main function to iterate through websites and save data to CSV
def main():
    all_scraped_data = []
    
    for website in websites:
        scraped_data = scrape_website(website)
        all_scraped_data.extend(scraped_data)
    
    # Save to CSV
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)  # Write the headers
        writer.writerows(all_scraped_data)  # Write the data

    print(f'Data successfully saved to {csv_file_path}')

if __name__ == '__main__':
    main()
