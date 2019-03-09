# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:57:41 2019

@author: shenshiyu
"""

import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from math import radians, cos, sin, asin, sqrt

time_start=time.time()

capacity = 5  
max_distance = 200000
min_arrival = 30000
max_vehicle = 5
max_detour = 1.5


"""
define get_distance function for coordinates
"""
#计算两点间距离
def get_distance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    dis=2*asin(sqrt(a))*6371*1000
    return dis

def region_filter(lng1,lat1,lng2,lat2,array):
# lng1,lat1: lower left conner
# lng2,lat2: upper right conner
    array = array[array[:,1]>lng1]
    array = array[array[:,1]<lng2]
    array = array[array[:,2]>lat1]
    array = array[array[:,2]<lat2]
    
    if len(array[1,:])>3:
        array = array[array[:,3]>lng1]
        array = array[array[:,3]<lng2]
        array = array[array[:,4]>lat1]
        array = array[array[:,4]<lat2]
        
    return array

"""
read files of passengers and stops
compute distance matrixs
find nearest board and alight stops
"""
passengers_file = pd.read_csv("passengers_file_100.csv")
stops_file = pd.read_csv("stops_file.csv")

passengers = passengers_file.values
stops = stops_file.values

lng1,lat1 = 106.615905,26.500245
lng2,lat2 = 106.752891,26.653767

passengers = region_filter(lng1,lat1,lng2,lat2,passengers)
stops = region_filter(lng1,lat1,lng2,lat2,stops)

distance_stops = np.zeros(shape = (len(stops[:,1]),len(stops[:,1])) )
for i in range(len(stops[:,1])):
    for j in range(len(stops[:,1])):
        distance_stops[i,j] = get_distance(stops[i,1],stops[i,2],stops[j,1],stops[j,2])
        
distance_home_stop = np.zeros(shape = (len(passengers[:,1]),len(stops[:,1])) )
for i in range(len(passengers[:,1])):
    for j in range(len(stops[:,1])):
        distance_home_stop[i,j] = get_distance(passengers[i,1],passengers[i,2],stops[j,1],stops[j,2])
home_near_stop = np.argmin(distance_home_stop,axis=1)

distance_company_stop = np.zeros(shape = (len(passengers[:,1]),len(stops[:,1])) )
for i in range(len(passengers[:,1])):
    for j in range(len(stops[:,1])):
        distance_company_stop[i,j] = get_distance(passengers[i,3],passengers[i,4],stops[j,1],stops[j,2])
company_near_stop = np.argmin(distance_company_stop,axis=1)


"""
creat initial routes
"""
routes = []  # define a route list
for i in range(len(passengers[:,1])):
    routes.append([ [home_near_stop[i]],[company_near_stop[i]],1 ])


"""
compute saving list
"""
# route i is outter, route j is inner

def save(i,j):
    d1 = distance_stops[routes[i][0][-1],routes[j][0][0]] 
    d2 = distance_stops[routes[j][0][0],routes[j][1][-1]] 
    d3 = distance_stops[routes[j][1][-1],routes[i][1][0]] 
    d4 = distance_stops[routes[i][0][0],routes[i][1][-1]]
    delta_d = d1 + d2 + d3 - d4
    return delta_d
    
list = [] # define a saving list
for i in range(1, len(passengers[:,1])): # for each node in graph
    for j in range(1, len(passengers[:,1])): # for each node in graph
        if i != j: # if 2 nodes are not the same 
            dist_save = save(i,j)
            # calculate the saving: sij = di0 + d0j − dij , for all i, j ≥ 1 and i != j;
            list.append([(i, j), dist_save]) # append the list with node combination and its saving

list.sort(key=lambda x: x[1], reverse=False)  # sort the list by saving


"""
check feasible merge
"""
def board_distance(route):
    board_dist = 0
    
    if len(route[0]) == 1:
        board_dist = 0
    else: 
        for i in range(1,len(route[0])):
            board_dist += distance_stops[route[0][i-1], route[0][i]] 
    
    return board_dist

def alight_distance(route):
    alight_dist = 0
    
    if len(route[1]) == 1:
        alight_dist = 0    
    else: 
        for j in range(1,len(route[1])):
            alight_dist += distance_stops[route[1][j-1], route[1][j]]
    
    return alight_dist

def route_distance(route):
  
    route_dist = board_distance(route) + alight_distance(route) + distance_stops[route[0][-1],route[1][0]]
    
    return route_dist    

def total_distance(route1,route2):
    original_dist = board_distance(route1) + alight_distance(route1) + route_distance(route2) 
    connect_dist = distance_stops[route1[0][-1],route2[0][0]] + distance_stops[route2[1][-1],route1[1][0]]
    total_dist = original_dist + connect_dist
    
    return total_dist

def arrival_distance(route1,route2):
    arrival_dist = alight_distance(route1) + alight_distance(route2) + distance_stops[route2[1][-1],route1[1][0]]
    
    return arrival_dist

def detour_ratio(route1,route2):
    detour = total_distance(route1,route2) / distance_stops[route1[0][0],route1[1][-1]]
    return detour
    
def feasible_merge(i, j, routes):

    outter = [] 
    inner = [] # start routes list and end routes list 
    for route in routes: # for each route in routes list
        if route[0][0] == home_near_stop[j] and route[1][-1] == company_near_stop[j]: # if a route start from j 
            inner = route # set the start route r
        elif route[0][-1] == home_near_stop[i] and route[1][0] == company_near_stop[i]: # if a route end with i
            outter = route # set the end route s

    
    if outter and inner and (outter[2] + inner[2] <= capacity) and (total_distance(outter,inner) <= max_distance) and (arrival_distance(outter,inner) <= min_arrival) and (detour_ratio(outter,inner) <= max_detour):
        return True # if the capacity satisfied, the routes could be merged 
    else: return False


"""
merge routes
"""
def merge_routes(i, j, routes):
    
    outter = [] 
    inner = [] 
    for route in routes: # for each route in routes list
        if route[0][0] == home_near_stop[j] and route[1][-1] == company_near_stop[j]: # if a route start from j 
            inner = route # set the start route r
        elif route[0][-1] == home_near_stop[i] and route[1][0] == company_near_stop[i]: # if a route end with i
            outter = route
            
    mix = []
#    if outter[0][-1] == inner[0][0]:
#        outter[0].remove(outter[0][-1])
#    if outter[1][0] == inner[1][-1]:
#        outter[1].remove(outter[1][0])
    
    mix.append(outter[0])
    mix[0].extend(inner[0])
    mix.append(inner[1])
    mix[1].extend(outter[1])
    mix.append(outter[2]+inner[2])
    
    routes.remove(outter) # remove r
    routes.remove(inner) # remove s
    routes.append(mix)


"""
mian process
"""

solution = routes.copy() # create initial routes - 1
listOfSaves = list # compute list of savings - 2
for save in listOfSaves: # 3
    i, j = save[0]
    if (feasible_merge(i, j, solution)):
        merge_routes(i, j, solution) # 5

solution.sort(key=lambda x: x[2], reverse=True)


"""
print result
"""
print('='*30)
print('total passengers',len(passengers[:,0]))
print('total stops',len(stops[:,0])) 
time_end=time.time()
print('time cost',time_end-time_start,'s')

print('='*30)   
for route in solution[0:max_vehicle]:
    for i in route[0]:
        print('{:.8},{:.8}'.format(stops[i,1],stops[i,2]))
    print('')
    for j in route[1]:
        print('{:.8},{:.8}'.format(stops[j,1],stops[j,2]))
    print('')
    print('{}={}'.format('passenger',route[2]))
    print('='*25)
    

"""
plot map
"""

map = Basemap(projection='mill',
            llcrnrlat = lat1-0.01,    # left corner latitude
            llcrnrlon = lng1-0.01,    # left corner longitude
            urcrnrlat = lat2+0.01,    # right corner latitude
            urcrnrlon = lng2+0.01,   # right corner longitude
            resolution='l', 
            area_thresh=100000)

fig = plt.figure(figsize=(20,20))
ax = plt.gca()

def plot_od_point(array):
    
    x1, y1 = map(array[:,1], array[:,2])
    x2, y2 = map(array[:,3], array[:,4])
    
    map.scatter(x1, y1, s=10, c='k', marker = 'o', alpha = 0.2)
    map.scatter(x2, y2, s=10, c='k', marker = '^', alpha = 0.2)  
    
def plot_stop(array):
    
    x1, y1 = map(array[:,1], array[:,2])
    map.scatter(x1, y1, s=5, c='k', marker = '+', alpha = 0.1) 

def plot_od_line(array):
    
    for i in range(len(array[:,1])):
        x3, y3 = map(np.append(array[i,1],array[i,3]), np.append(array[i,2],array[i,4]))
        map.plot(x3, y3, c='k',alpha = 0.1,linewidth = 1)

def plot_route(route_no):
    
    for k in range(route_no):
        
        lon1,lat1,lon2,lat2 = [],[],[],[]
        color = random.choice('rgbcmky')
        
        for i in solution[k][0]:
            lon1.append(stops[i,1])
            lat1.append(stops[i,2])
            
        for i in solution[k][1]:
            lon2.append(stops[i,1])
            lat2.append(stops[i,2])
              
        lon1 = np.array(lon1)
        lat1 = np.array(lat1)
        lon2 = np.array(lon2)
        lat2 = np.array(lat2)
        
        x1, y1 = map(lon1, lat1)
        x2, y2 = map(lon2, lat2)
        x3, y3 = map(np.append(lon1,lon2), np.append(lat1,lat2))
        
        map.scatter(x1, y1, s=50, c=color, marker = 'o')
        map.scatter(x2, y2, s=50, c=color, marker = '^')   
        map.plot(x3, y3, c=color)

plot_od_point(passengers)
plot_od_line(passengers)
plot_stop(stops)
plot_route(5)









