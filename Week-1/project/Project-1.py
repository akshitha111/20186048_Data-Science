#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as pl
import pandas as pd
import time
import requests
import urllib.request
from bs4 import BeautifulSoup
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")


# In[2]:


with urllib.request.urlopen('https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_1970') as response:
   wiki_chart = response.read()
soup = BeautifulSoup(wiki_chart, "lxml")
tables = soup.find('table', attrs={'class':'wikitable sortable'})


songList = []
tr_list = tables.find_all('tr')
for tr in tr_list:
    td_list = tr.find_all('td')
    if td_list == [] :
        td_list = []
    else : 
        ranking = td_list[0].get_text()
        title = td_list[1].get_text()
        band_singer = td_list[2].get_text()
        soup_of_link = BeautifulSoup(str(td_list), "lxml")
        url = td_list[2].a["href"]
        dict_entry = {'band_singer' : band_singer,
        'ranking' : ranking,
        'title' : title,
        'url' : url}
        songList.append(dict_entry)


# In[3]:


songList[2:4]


# In[4]:


yearstext = {}
for year in range(1992, 2015):
    with urllib.request.urlopen(f'https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_%s' % year) as response:
        year_text = {year : response.read()}
        yearstext.update(year_text)
        time.sleep(1)
        


# In[5]:


def parse_year(the_year, yeartext_dict):
    year = the_year
    yearinfo = []
    song = []
    songurl = []
    band_singer = []
    title = []
    url = []
    title_text = ''
    i = 0
    title_string = ''
    band_singer = ''
    soup = BeautifulSoup(yearstext[year], "lxml")
    tables = soup.find('table', attrs={'class':'wikitable sortable'})        
    #iterates through tree structure, scraping our data
    tr_list = tables.find_all('tr')    
    for tr in tr_list:
        td_list = tr.find_all('td')
        if td_list == [] :
            td_list = []
        else : 
            ranking = tr.th.string
            links = tr.td.findAll('a')
            number_of_links = len(links)   
            if number_of_links == 0:
                songurl = [None]
                title_text = [a['title']]
                song.append(a['title'] )
            else :
                i = 0
                for a in tr.td.findAll('a') : 
                    song.append(a['title'] )
                    title_string = '\"' + a['title'] + '\"'    
                    if i == 0 :
                        title_text = title_string
                        i = i + 1
                    else :
                        title_text = title_text + ' / ' + title_string
                        i = i + 1    
                    songurl.append(a['href'])
            title = song
            #finds next td tag
            tr.td.findNext('td') 
            temp = len(tr.td.findNext('td').findAll('a'))
            if temp == 0:
                singer_url = [None]
                band_singer = tr.td.findNext('td').string
                band_singer = [band_singer]
            elif temp == 1:
                singer_url = tr.td.findNext('td').a['href']
                singer_url = [singer_url]
                band_singer = tr.td.findNext('td').a.string
                band_singer = [band_singer]
            else:
                singer_url = []
                band_singer = []
                for a in tr.td.findNext('td').findAll('a'):
                    webpage = a['href']
                    singer_url.append(webpage)
                    band_singer.append(a.string)            
            #creates dictionary entry                   
            dict_entry = {'band_singer' : band_singer,
            'ranking' : ranking,
            'song' : title, 'songurl': songurl, 'titletext' : title_text,
            'url' : singer_url}
            #appends new dictionary to our list
            yearinfo.append(dict_entry)      
            songurl = []
            song = []
            title_string = ''
            title_text = ''    
    return(yearinfo)


# In[6]:


years_dictionary = {}
for year in range(1992, 2015):
    years_dictionary.update({year : parse_year(year, yearstext)})
    


# In[7]:


year_info = years_dictionary  


# In[8]:


parse_year(1997, yearstext)[:5]


# In[9]:


import json


# In[10]:


fd = open("year_info.json","w+")
json.dump(year_info, fd)
fd.close()
del year_info


# In[11]:


with open("year_info.json", "r") as fd:
    year_info = json.load(fd)


# In[12]:


rows = []
for year in year_info.keys():
    for song in year_info[year]:
        song['year'] = year
        rows.append(song)  


# In[13]:


rows


# In[14]:


band_singer = []
songurl = ''
title_text = ''
singer_url = []
starting_length = len(rows)
for dics in rows: 
    if starting_length == 0:
        break
    dict_add = {}   
    # checks if our dictionary contains lists longer than one element
    if len(dics['band_singer']) > 1:
        i = 0
        j = len(dics['band_singer'])
        
        for value in dics['band_singer']:
            # new dictionary entry to append to our list
            dict_add = {'band_singer' : dics['band_singer'][i],
            'ranking' : dics['ranking'],
            'song' : dics['song'], 'songurl': dics['songurl'], 'titletext' : dics['titletext'],
            'url' : dics['url'][i], 'year' : dics['year']}
            rows.append(dict_add)
            i = 1 + i
            j = j - 1
    starting_length = starting_length - 1

rows2 = []
band_singer = []

# loops through our list again, removing duplicate entries
for dics in rows:
    if len(dics['band_singer']) == 1 or len(dics['band_singer']) > 5:
        rows2.append(dics)

# turns all remaining one element lists into strings
for row in rows2:
    for key in row:
        row[key] = str(row[key])
        row[key] = row[key].strip("[]")
        row[key] = row[key].strip("''")
        


# In[15]:


rows2


# In[16]:


flat_Frame = pd.DataFrame(rows2)
flat_Frame['year'].dtype


# In[17]:


flat_Frame


# In[18]:


flat_Frame['year'] = flat_Frame['year'].astype(np.uint16)


# In[19]:


artist_count = flat_Frame["band_singer"].value_counts()


# In[20]:


artist_count[:8].plot.barh()


# In[21]:


scored_df = pd.DataFrame(np.array(flat_Frame[['band_singer', 'song', 'ranking']]))
scored_df.columns = ['band_singer', 'song', 'ranking']
# scored_df.groupby(['band_singer', 'song'])['ranking']

scored_df['ranking'] = scored_df['ranking'].apply(lambda x: 101 - int(x))

#display(scored_df)

scored_df.groupby('band_singer')['ranking'].aggregate(np.sum).sort_values(ascending=False).head(20).plot.barh()


# In[22]:


"""
While Rihanna keeps her place at the head of the pack (by a sizeable margin, it should be noted), 
Mariah Carey gains materially over R. Kelly, Ludacris and Lil Wayne. The reason for this is that while
Mariah's songs appear in the top 100 less frequently than those other three, the latter chart weighs
the relative rank of her songs while the former disregards this and looks at only frequency. Including
relative rank also gives Katy Perry a place near the top, too.
"""


# In[23]:


url_cache = {}


# In[24]:


def get_page(url):
    # Check if URL has already been visited.
    if (url not in url_cache) or (url_cache[url]==1) or (url_cache[url]==2):
        time.sleep(1)
        # try/except blocks are used whenever the code could generate an exception (e.g. division by zero).
        # In this case we don't know if the page really exists, or even if it does, if we'll be able to reach it.
        try:
            r = requests.get("http://en.wikipedia.org%s" % url)

            if r.status_code == 200:
                url_cache[url] = r.text
            else:
                url_cache[url] = 1
        except:
            url_cache[url] = 2
    return url_cache[url]


# In[25]:


# sort the flatframe by year
flat_Frame = flat_Frame.sort_values('year')


# In[26]:


flat_Frame["url"].apply(get_page)


# In[29]:


print("Number of bad requests:",np.sum([(url_cache[k]==1) or (url_cache[k]==2) for k in url_cache])) # no one or 0's)
print("Did we get all urls?", len(flat_Frame.url.unique())==len(url_cache)) # we got all of the urls


# In[33]:


with open("artistinfo.json","w") as fd:
    json.dump(url_cache, fd)
del url_cache


# In[34]:


with open("artistinfo.json") as json_file:
    url_cache = json.load(json_file)


# In[35]:


def singer_band_info(url, page_text):
    bday_dict = {}
    bday = ''
    ya = ''
    # soupify our webpage
    soup = BeautifulSoup(page_text[url], "lxml")
    tables = soup.find('table', attrs={'class':'infobox vcard plainlist'})
    if (tables == None):
        tables = soup.find('table', attrs={'class':'infobox biography vcard'})
    bday = tables.find('span', {'class': 'bday'})
    if bday: 
        bday = bday.get_text()[:4]
        bday_dict = {'url' : url, 'born' : bday, 'ya' : ya}
    else:
        ya = False
        for tr in tables.find_all('tr'):
            if hasattr(tr.th, 'span'):
                if hasattr(tr.th.span, 'string'):
                    if tr.th.span.string == 'Years active':                
                        if hasattr(tr.th, 'span'):
                            #ya = tr.td.string
                            ya = tr.td.text   #DK add
                            bday = 'False'
                            bday_dict = {'url' : url, 'born' : 'False', 'ya' : ya}
    return(bday_dict)


# In[36]:


url = '/wiki/Iggy_Azalea'
singer_band_info(url, url_cache)


# In[37]:


list_of_addit_dicts = []
bday_dict = {}
for url in url_cache.keys():   
    try:
        bday_dict = singer_band_info(url, url_cache)
        list_of_addit_dicts.append(bday_dict)
    except:
        break


# In[38]:


list_of_addit_dicts


# In[39]:


additional_df = pd.DataFrame(list_of_addit_dicts)


# In[41]:


largedf = pd.merge(flat_Frame, additional_df, left_on='url', right_on='url', how="outer")


# In[42]:


largedf


# In[43]:


new_df = pd.DataFrame(np.array(largedf[['year', 'band_singer', 'ranking', 'born']]))
new_df.columns = ['year', 'band_singer', 'ranking', 'born']
new_df['born'] = pd.to_numeric(new_df['born'], errors='coerce').fillna(0).astype(np.int64)                     #convert to int
new_df['year_minus_born'] = new_df['year'] - new_df['born']
sorted_df = new_df.sort_values(['band_singer', 'ranking']);
filtered_df = sorted_df.drop_duplicates(subset='band_singer', keep='first')
filtered_df = filtered_df.query('born > 0')
print(filtered_df)
filtered_df.groupby('band_singer')['year_minus_born'].aggregate(np.sum).sort_values(ascending=False).plot.hist(bins=10)


# In[44]:


new_df_2 = pd.DataFrame(np.array(largedf[['year', 'band_singer', 'ranking', 'ya']]))
new_df_2.columns = ['year', 'band_singer', 'ranking', 'ya']
new_df_2['ya'] = new_df_2['ya'].str[:4]
new_df_2['ya'] = pd.to_numeric(new_df_2['ya'], errors='coerce').fillna(0).astype(np.int64)                     #convert to int
new_df_2['year_minus_ya'] = new_df_2['year'] - new_df_2['ya']
sorted_df_2 = new_df_2.sort_values(['band_singer', 'ranking']);
filtered_df_2 = sorted_df_2.drop_duplicates(subset='band_singer', keep='first')
filtered_df_2 = filtered_df_2.query('ya > 0')
filtered_df_2 = filtered_df_2.query('year_minus_ya < 40')
print(filtered_df_2)
filtered_df_2.groupby('band_singer')['year_minus_ya'].aggregate(np.sum).sort_values(ascending=False).plot.hist(bins=10)


# In[ ]:




