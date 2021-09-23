# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:42:20 2021

@author: Marbalza
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher
from datetime import date

today = date.today()

dateStr = today.strftime("%Y-%m-%d")

ident = playerid_lookup('carrasco', 'carlos')
player = ident['key_mlbam'].values
player = player.item()

work_df = statcast_pitcher('2021-01-16', dateStr, player)


# RMSE function
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets).astype('double') ** 2).mean())

# Initializing arrays, lists & variables needed later.
# The colors array keeps the code for each color needed por every different kind of pitch
colors = np.array(['r', 'b', 'k', 'c', 'm', 'y', 'g', 'burlywood', 'chartreuse'])

#The rp_avg variables will store the centroids por each type of pitch
rp_avg_X = 0.0
rp_avg_Z = 0.0
local = 0

# Read data file with release points from Baseball Savant 
# Change the path and name of the CSV file to where you have it stored
#work_df = pd.read_csv("E:/Work/data/ck.csv")

# Drop the rows with nan release points info

work_df.dropna(subset = ["release_pos_x"], inplace=True)

#Mix of pitches (types, quantity)
number_column = work_df.loc[:,'pitch_type']
pitches = number_column.values
pitches = np.unique(pitches)
pitches_df = pd.DataFrame(pitches, columns = ['pitch_type'])
pitches_df['rp_avg_x'] = 0
pitches_df['rp_avg_z'] = 0
pitches_df['color'] = 'mediumseagreen'
lenght_mix = len(pitches)


# Creating and sorting the results dataframe, rp_df
rp_df = work_df[['pitch_type', 'release_pos_x', 'release_pos_z']].copy()
rp_df['rp_avg_x'] = 0
rp_df['rp_avg_z'] = 0
rp_df['color'] = 'mediumseagreen'

rp_df['pitch_type_doux'] = pd.Categorical(
    rp_df['pitch_type'], 
    categories=pitches, 
    ordered=True
)
rp_df.sort_values('pitch_type_doux', inplace=True)
rp_df = rp_df.drop(columns=['pitch_type_doux'])
rp_df.reset_index(drop=True, inplace=True)
count = len(rp_df)


#Calculating centroids for every type of pitch and adding the column (plus the color column)
for i in range(lenght_mix):
    for j in range(count):
        if rp_df.iloc[j]['pitch_type']==pitches[i]:
            rp_avg_X += rp_df.iloc[j]['release_pos_x']
            rp_avg_Z += rp_df.iloc[j]['release_pos_z']
            if pitches[i]=="CH":
                tone=colors[0]
            elif pitches[i]=="SL":
                tone=colors[1]
            elif pitches[i]=="FF":
                tone=colors[2]
            elif pitches[i]=="FC":
                tone=colors[3]
            elif pitches[i]=="CU":
                tone=colors[4]
            elif pitches[i]=="FS":
                tone=colors[5]
            elif pitches[i]=="KC":
                tone=colors[6]
            elif pitches[i]=="SI":
                tone=colors[7]
            elif pitches[i]=="CS":
                tone=colors[8]
            
            rp_df.at[j,"color"] = tone
            local += 1
            
    #rp_df_new = rp_df_new.append(rp_df.query('pitch_type == @a'))    
    rp_df.dropna(subset = ["pitch_type"], inplace=True)
    rp_avg_X = rp_avg_X/(local)
    rp_avg_Z = rp_avg_Z/(local)
    
    rp_df['rp_avg_x'].mask(rp_df['pitch_type']==pitches[i], rp_avg_X, inplace=True)
    rp_df['rp_avg_z'].mask(rp_df['pitch_type']==pitches[i], rp_avg_Z, inplace=True)
    
    pitches_df['rp_avg_x'].mask(pitches_df['pitch_type']==pitches[i], rp_avg_X, inplace=True)
    pitches_df['rp_avg_z'].mask(pitches_df['pitch_type']==pitches[i], rp_avg_Z, inplace=True)
    a = pitches[i]
    z = rp_df.query('pitch_type == @a')
    z = z['color'].values[0]
    pitches_df['color'].mask(pitches_df['pitch_type']==pitches[i], z, inplace=True)    
    rp_avg_X = 0
    rp_avg_Z = 0 
    local = 0
    

# Configuring axes and grid for plot


plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
fig, ax1 = plt.subplots()

ax1.yaxis.set_ticks_position("both")
plt.grid(True)
ax1.axis('equal')
ax1.set(xlim=(-6, 6), ylim=(0, 8))

# Title and Subtitle
title=work_df.loc[0,'player_name']
pre_name=title.split(", ")
title=pre_name[1] + " " + pre_name[0] + "'s RPs - " + str(len(rp_df['color'])) + " pitches"

plt.suptitle(title, y=1.05, fontsize=18)
plt.title("Axes are in feet, catcher's point of view", fontsize=10)

#Plot individual pitches
ax1.scatter(rp_df['release_pos_x'], rp_df['release_pos_z'], c=rp_df['color'], alpha=0.05, s=20, marker='.')   


#Plot Centroids and Labels
number_column = pitches_df.loc[:,'pitch_type']
pitch = number_column.values
number_column = pitches_df.loc[:,'rp_avg_x']
avg_x = number_column.values
number_column = pitches_df.loc[:,'rp_avg_z']
avg_z = number_column.values
number_column = pitches_df.loc[:,'color']
color = number_column.values



for i in range(len(pitch)):
    a = pitches[i]
    z = rp_df.query('pitch_type == @a')
    x = rmse(z['release_pos_x'], z['rp_avg_x'])
    y = rmse(z['release_pos_z'], z['rp_avg_z'])
    
    xx = mae(z['rp_avg_x'], z['release_pos_x'])
    yy = mae(z['rp_avg_z'], z['release_pos_z'])
    
    mega = pitch[i] + " RMSEx= {:0.3f}".format(x) + " , RMSEy= {:0.3f}".format(y) + " / MAEx= {:0.3f}".format(xx) + " , MAEy= {:0.3f}".format(yy) + " - {:0.0f}".format(len(z['release_pos_x'])) + " pitch(es)"
    ax1.scatter(avg_x[i], avg_z[i], c=color[i], alpha=1, s=40, marker='x', label=mega)
    print(mega)

plt.legend(markerscale=1, loc='lower right', borderpad=0.25, prop={"size":7.6})


avgx_mean = pitches_df['rp_avg_x'].mean()

if pitches_df['rp_avg_x'].mean()<0:
    locat = 1
else:
    locat = 2
    
if avgx_mean > -0.8 and avgx_mean < 0.8:
    locat = 5

minx = rp_df['release_pos_x'].min()
maxx = rp_df['release_pos_x'].max()
miny = rp_df['release_pos_z'].min()
maxy = rp_df['release_pos_z'].max()

factor=0.0
longx = abs(maxx)-abs(minx)
longy = abs(maxy)-abs(miny)

if longx >= 2:
    factor = 1.10
elif longx < 0:
    factor = 1.25
else:
    factor = 2
    
if pitches_df['rp_avg_x'].mean()>-0.5 and pitches_df['rp_avg_x'].mean()<0.5:
    factor = 0.27
elif abs(pitches_df['rp_avg_x'].mean()) > 1.75 and longy < 1.75:
    factor = 2

if abs(minx)<0.5 or abs(maxx)<0.5:
    factor = 0.9

delta=1


if (minx < 0 and maxx) < 0 or (minx > 0 and maxx > 0):
    delta = 1.158

#delta=1.1

zoom = -2.08654*(abs(abs(maxx)-abs(minx)))+7.63442*delta - abs(minx)/factor
    
axins = zoomed_inset_axes(ax1, zoom, loc=locat) 
axins.scatter(rp_df['release_pos_x'], rp_df['release_pos_z'], c=rp_df['color'], alpha=0.15, s=20, marker='.')

for i in range(len(pitch)):
    a = pitches[i]
    z = rp_df.query('pitch_type == @a')
    x = rmse(z['release_pos_x'], z['rp_avg_x'])
    y = rmse(z['release_pos_z'], z['rp_avg_z'])
    
    xx = mae(z['rp_avg_x'], z['release_pos_x'])
    yy = mae(z['rp_avg_z'], z['release_pos_z'])
    
    mega = pitch[i] + " RMSEx= {:0.3f}".format(x) + " , RMSEy= {:0.3f}".format(y) + " / MAEx= {:0.3f}".format(xx) + " , MAEy= {:0.3f}".format(yy) + " - {:0.0f}".format(len(z['release_pos_x'])) + " pitch(es)"
    axins.scatter(avg_x[i], avg_z[i], c=color[i], alpha=1, s=80, marker='x', label=mega)



axins.set_xlim(minx, maxx) # Limit the region for zoom
axins.set_ylim(miny, maxy)

plt.xticks(visible=False)  # Not present ticks
plt.yticks(visible=False)
#
## draw a bbox of the region of the inset axes in the parent axes and
## connecting lines between the bbox and the inset axes area
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.7")

plt.draw()
plt.show()


#######################################################
"""
work_df['game_date'] = pd.to_datetime(work_df.game_date)
work_df = work_df.sort_values(by='game_date')

work_df = work_df[work_df.pitch_type=='FS']

pre = work_df.groupby('game_date').std()
x = pd.Series(work_df['game_date'].unique())

s = pd.Series(pre['release_pos_x'])

y = s.rolling(3).mean()

ER = pd.Series([12.2,9.4,16.7,13.0,19.8,14.7,14.4,18.8,13.8,18.6,22.2,19.6,14.1,18.6,19.0,7.8,14.1,15.7,10.0,5.6,12.0,16.5,11.5,16.0,11.0,12.1,17.1,19.4,16.3,16.0])


"""
s = pd.Series([5, 5, 6, 7, 5, 5, 5])
x = pd.Series([1, 2, 3, 4, 5, 6, 7])
s2 = s.rolling(5).std()

"""
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
fig, ax2= plt.subplots()

ax2.yaxis.set_ticks_position("both")
plt.grid(True)
#ax1.axis('equal')
#ax2.set(ylim=(0.05, 0.095))


# Title and Subtitle
#title=work_df.loc[0,'player_name']
#pre_name=title.split(", ")
#title=pre_name[1] + " " + pre_name[0] + "'s RPs - " + str(len(rp_df['color'])) + " pitches"

#plt.suptitle(title, y=1.05, fontsize=18)
#plt.title("Axes are in feet, catcher's point of view", fontsize=10)

#Plot individual pitches
ax2.scatter(x,y) 
#ax2.scatter(x,s) 
plt.plot(x,y)

ax3=ax2.twinx()
#ax3.set(ylim=(-1, 7))
ax3.scatter(x,ER,c='k')

plt.plot(x,ER,color="black")

plt.show()
"""