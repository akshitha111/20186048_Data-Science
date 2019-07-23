#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests


# In[2]:


import requests


# In[3]:


req = requests.get("https://en.wikipedia.org/wiki/Harvard_University")


# In[5]:


req


# In[6]:


type(req)


# In[7]:


dir(req)


# In[8]:


page = req.text
page


# In[9]:


from bs4 import BeautifulSoup


# In[10]:


from bs4 import BeautifulSoup


# In[11]:


soup = BeautifulSoup(page, 'html.parser')


# In[12]:


soup


# In[13]:


type(page)


# In[14]:


type(soup)


# In[22]:


#print (soup.prettify())"""


# In[17]:


soup.title


# In[18]:


"title" in dir(soup)


# In[26]:


soup.h1


# In[31]:


x = soup.find_all("p")


# In[32]:


x[1]


# In[33]:


x[0]


# In[34]:


len(soup.find_all("p"))


# In[35]:


soup.table["class"]


# In[36]:


[t["class"] for t in soup.find_all("table") if t.get("class")]


# In[37]:


my_list = []
for t in soup.find_all("table"):
    if t.get("class"):
        my_list.append(t["class"])
my_list


# In[103]:


table_html = str(soup.findAll("table", "wikitable")[2])


# In[104]:


from IPython.core.display import HTML

HTML(table_html)


# In[118]:


rows = [row for row in soup.findAll("table", "wikitable")[2].find_all("tr")]
rows


# In[119]:


rem_nl = lambda s: s.replace("\n", " ")


# In[107]:


def power(x, y):
    return x**y

power(2, 3)


# In[108]:


def print_greeting():
    print ("Hello!")
    
print_greeting()


# In[109]:


def get_multiple(x, y=1):
    return x*y

print ("With x and y: ", get_multiple(10, 2))
print ("With x only: ", get_multiple(10))


# In[110]:


def print_special_greeting(name, leaving=False, condition="nice"):
    print ("Hi", name)
    print ("How are you doing in this", condition, "day?")
    if leaving:
        print ("Please come back!")


# In[111]:


print_special_greeting("John")


# In[112]:


print_special_greeting("John", True, "rainy")


# In[113]:


print_special_greeting("John", True)


# In[114]:


print_special_greeting("John", condition="horrible")


# In[115]:


def print_siblings(name, *siblings):
    print (name, "has the following siblings:")
    for sibling in siblings:
        print (sibling)
    print
        
print_siblings("John", "Ashley", "Lauren", "Arthur")
print_siblings("Mike", "John")
print_siblings("Terry")


# In[120]:


columns = [rem_nl(col.get_text()) for col in rows[0].find_all("th") if col.get_text()]
columns


# In[121]:


indexes = [rem_nl(row.find("th").get_text()) for row in rows[1:]]
indexes


# In[122]:


HTML(table_html)


# In[123]:


to_num = lambda s: s[-1] == "%" and int(s[:-1]) or None


# In[135]:


values = [to_num(value.get_text()) for row in rows[1:] for value in row.find_all("td")]
values


# In[126]:


stacked_values = zip(*[values[i::3] for i in range(len(columns))])
list(stacked_values)


# In[136]:


HTML(table_html)


# In[138]:


def print_args(arg1, arg2, arg3):
    print (arg1, arg2, arg3)


print_args(1, 2, 3)


print_args([1, 10], [2, 20], [3, 30])


# In[139]:


parameters = [100, 200, 300]

p1 = parameters[0]
p2 = parameters[1]
p3 = parameters[2]

print_args(p1, p2, p3)


# In[140]:


p4, p5, p6 = parameters

print_args(p4, p5, p6)


# In[141]:


print_args(*parameters)


# In[142]:


print_args(*parameters)


# In[144]:


import pandas as pd


# In[145]:


df = pd.DataFrame(stacked_values, columns=columns, index=indexes)
df


# In[146]:


HTML(table_html)


# In[148]:


stacked_values = zip(*[values[i::3] for i in range(len(columns))])
pText = pd.DataFrame(stacked_values, columns=columns, index=indexes)
pText


# In[149]:


columns = [rem_nl(col.get_text()) for col in rows[0].find_all("th") if col.get_text()]
stacked_values = zip(*[values[i::3] for i in range(len(columns))])
data_dicts = [{col: val for col, val in zip(columns, col_values)} for col_values in stacked_values]
data_dicts


# In[150]:


pd.DataFrame(data_dicts, index=indexes)


# In[151]:


stacked_colValues = [values[i::3] for i in range(len(columns))]
stacked_colValues


# In[152]:


data_lists = {col: val for col, val in zip(columns, stacked_colValues)}
data_lists


# In[153]:


pd.DataFrame(data_lists, index=indexes)


# In[154]:


pText.dtypes


# In[155]:


pText.dropna()


# In[156]:


pText.dropna(axis=1)


# In[164]:


df_clean = pText.fillna(0).astype(int)
pTextOpti


# In[165]:


df_clean.dtypes


# In[166]:


df_clean.describe()


# In[160]:


import numpy as nump


# In[167]:


df_clean.values


# In[171]:


nump.mean(df_clean.Undergrad)


# In[172]:


nump.std(df_clean)


# In[195]:


df_clean["Undergrad"]


# In[196]:


df_clean.iloc[0]


# In[203]:


df_clean.loc["Asian/Pacific Islander "]


# In[204]:


df_clean.ix[0]


# In[210]:


df_clean.iloc[3, 1]


# In[212]:


df_flat = df_clean.stack().reset_index()
df_flat.columns = ["race", "source", "percentage"]
df_flat


# In[213]:


grouped = df_flat.groupby("race")
grouped.groups


# In[214]:


type(grouped)


# In[215]:


mean_percs = grouped.mean()
mean_percs


# In[216]:


type(mean_percs)


# In[218]:


for name, group in df_flat.groupby("source", sort=True):
    print (name)
    print (group)


# In[219]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[220]:


mean_percs.plot(kind="bar");


# In[ ]:




