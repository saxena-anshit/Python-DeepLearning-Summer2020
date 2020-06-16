# Author: Anshit Saxena - 16292998
# Python script to scrape webpages for titles and links

# to import required modules
import requests
from bs4 import BeautifulSoup
import os

# use request with get function to assign the web page content to htmlPage variable
htmlPage = requests.get("https://en.wikipedia.org/wiki/Deep_learning")
# parsing the page using beautiful soup
bsObj = BeautifulSoup(htmlPage.content, "html.parser")
print("Title is: ", bsObj.title.string)
print("links in ", bsObj.title.string, ": ")

list = []
# Creating a list of all links
for link in bsObj.find_all('a'):
    # print(link.get('href'))
    list.append(link.get('href'))
print(list)

# writing the list with all links to the file
with open('listfile.txt', 'w') as filehandle:
    for listitem in list:
        filehandle.write('%s\n' % listitem)