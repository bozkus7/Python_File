import pandas as pd
import numpy as np
import re
import dateparser
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use('ggplot') 

def read_file(file):
    x = open(file, 'r', encoding = 'utf-8')
    y = x.read()
    content = y.splitlines()
    return content

chat = read_file('chat.txt')
##print("length of chat is:")
##print(len(chat))

clean_chat = [line for line in chat if not "joined using this" in line]

clean_chat = [line for line in clean_chat if len(line) > 1]
##print("length of clean_chat is:")
##print(len(clean_chat))

#Clean out the left notification lines
clean_chat = [line for line in clean_chat if not line.endswith("left")]
#print(len(clean_chat))

"""
Flow:
For every line, see if it matches the expression which is starting with the format "number(s)+slash" eg "12/"
If it does, it is a new line of conversion as they begin with dates, add it to msgs container
Else, it is a continuation of the previous line, add it to the previous line and append to msgs,
then pop previous line.
"""

msgs = []
pos = 0

for line in clean_chat:
    if re.findall("\A\d+[/]", line):
        msgs.append(line)
        pos += 1
    else:
        take = msgs[pos-1] + ". " + line
        msgs.append(take)
        msgs.pop(pos-1)

##print(len(msgs))

#Next, we will need to extract Date, Time, Name and Message Content from our msgs data using the codes below:

time = [msgs[i].split(',')[1].split('-')[0] for i in range(len(msgs))]
time = [s.strip(' ') for s in time] # Remove spacing
##print("length of time is:")
##print(len(time))

date = [msgs[i].split(',')[0] for i in range(len(msgs))]
name = [msgs[i].split('-')[1].split(':')[0] for i in range(len(msgs))]
        
content = []
for i in range(len(msgs)):
  try:
    content.append(msgs[i].split(':')[2])
  except IndexError:
    content.append('Missing Text')
len(content)

df = pd.DataFrame(list(zip(date, time, name, content)), columns = ['Date', 'Time', 'Name', 'Content'])
df = df[df["Content"]!='Missing Text']
df.reset_index(inplace=True, drop=True)

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

df['weekday'] = df['DateTime'].apply(lambda x: x.day_name())
df['Letter_Count'] = df['Content'].apply(lambda s : len(s))
df['Word_Count'] = df['Content'].apply(lambda s : len(s.split(' ')))

df['Hour'] = df['Time'].apply(lambda x : x.split(':')[0]) 
# The first token of a value in the Time Column contains the hour (Eg., "12" in "12:15")

#saving this to csv format
df.to_csv("whatsappChat.csv")






