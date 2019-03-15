# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:39:01 2019

@author: shenshiyu
"""
import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from math import radians, cos, sin, asin, sqrt

"""
Define Classes

1 - Passenger
2 - Stop
3 - Route
4 - Graph contains passengers\stops\routes

"""
class Passenger:
    
    def __init__(self, index, home_x, home_y, company_x, company_y):
        self.index = index # new index of filtered files
        self.home_x = home_x 
        self.home_y = home_y
        self.company_x = company_x 
        self.company_y = company_y
        
        self.distance_home_stop = {}
        self.distance_company_stop = {}
        
        self.home_near_stops = []
        self.company_near_stops = []
        
        self.home_stop = 0
        self.company_stop = 0
    
    def __str__(self):
        return '%s' % self.index
    __repr__ = __str__

class Stop:
    
    def __init__(self, index, x, y):
        self.index = index # new index of filtered files
        self.x = x
        self.y = y
        
        self.distance_stop_stop = {}

    def __str__(self):
        return '%s' % self.index
    __repr__ = __str__
        
class Route:
    
    def __init__(self, board_stop, alight_stop, passenger):
        self.board_stop = board_stop
        self.alight_stop = alight_stop
        self.passengers = passenger
        self.demand = 1
        self.board_dist = 0
        self.alight_dist = 0
        self.distance = 0
        self.profit = 0
        
    def __str__(self):         
        return '%s' % [self.board_stop,self.alight_stop,self.passengers,self.demand,self.profit]
    __repr__ = __str__

    def cal_board_dist(self):
    # compute board distance of a route    
        self.board_dist = 0
        if len(self.board_stop) > 1:
            for i in range(1,len(self.board_stop)):
                self.board_dist += self.board_stop[i-1].distance_stop_stop[self.board_stop[i]]
        
        return self.board_dist
    
    def cal_alight_dist(self):
    # compute alight distance of a route     
        self.alight_dist = 0
        if len(self.alight_stop) > 1:
            for i in range(1,len(self.alight_stop)):
                self.alight_dist += self.alight_stop[i-1].distance_stop_stop[self.alight_stop[i]]  
        
        return self.alight_dist
    
    def cal_distance(self):
    # compute total distance of a route     
        self.distance = self.cal_board_dist() + self.cal_alight_dist() + self.board_stop[-1].distance_stop_stop[self.alight_stop[0]]   
        
        return self.distance

    def cal_profit(self):
    # compute total profit of a route    
        total_cost = cost * self.cal_distance()
        
        total_dist = 0
        for passenger in self.passengers:
            total_dist += passenger.home_stop.distance_stop_stop[passenger.company_stop]
        total_revenue = fare * total_dist
        
        self.profit = total_revenue - total_cost
    
class Graph:
    
    def __init__(self,num_passengers,num_stops):
        self.num_passengers = num_passengers
        self.num_stops = num_stops
        self.passengers = []
        self.stops = []
        self.routes = []
        
    def add_passenger(self, passenger):
    # add passengers to the graph list
        self.passengers.append(passenger)

    def add_stop(self, stop):
    # add stops to the graph list
        self.stops.append(stop)
        
    def add_route(self, route):
    # add routes to the graph list
        self.routes.append(route)

    def cal_distance(self):
    # compute distances between stops, home & stops, company & stops
        for stop1 in self.stops:
            for stop2 in self.stops:
                stop1.distance_stop_stop[(stop2)] = get_distance(stop1.x,stop1.y,stop2.x,stop2.y)

        for passenger in self.passengers:
            for stop in self.stops:
                passenger.distance_home_stop[(stop)] = get_distance(passenger.home_x,passenger.home_y,stop.x,stop.y)
                
        for passenger in self.passengers:
            for stop in self.stops:
                passenger.distance_company_stop[(stop)] = get_distance(passenger.company_x,passenger.company_y,stop.x,stop.y)
    
    def get_near_stops(self):
    # find avaliable stops near home & company for passengers
        for passenger in self.passengers:
            for stop in self.stops:
                if passenger.distance_home_stop[stop] <= min_walk_dist:
                    passenger.home_near_stops.append(stop)
            if passenger.home_near_stops == []:
                passenger.home_near_stops.append(min(passenger.distance_home_stop, key=passenger.distance_home_stop.get))

        for passenger in self.passengers:
            for stop in self.stops:
                if passenger.distance_company_stop[stop] <= min_walk_dist:
                    passenger.company_near_stops.append(stop)
            if passenger.company_near_stops == []:
                passenger.company_near_stops.append(min(passenger.distance_company_stop, key=passenger.distance_company_stop.get))

    def sort_routes(self):
    # sort the routes by profit
        for route in self.routes:
            route.cal_profit()
        self.routes.sort(key=lambda x: x.profit, reverse=True)

"""
Define Global Functions

1 - define distance function
2 - filter the region by coordinates
3 - read from files
4 - initial the 4 classes

"""
def get_distance(lng1,lat1,lng2,lat2):
# define distance function for coordinates calculation    
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    dis=2*asin(sqrt(a))*6371
        
    return dis

def region_filter(lng1,lat1,lng2,lat2,array):
# define region filter function for passengers and stops in a certain area       
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

def read(p_file_name, s_file_name):
# read files of passengers and stops
    passengers_file = pd.read_csv(p_file_name)
    stops_file = pd.read_csv(s_file_name)
    passengers = passengers_file.values
    stops = stops_file.values
    # filter region
    passengers = region_filter(lng1,lat1,lng2,lat2,passengers)
    stops = region_filter(lng1,lat1,lng2,lat2,stops)
    # compute filtered number of passengers and stops
    num_passengers = len(passengers[:,0])
    num_stops = len(stops[:,0])
    # initial graph
    graph = Graph(num_passengers, num_stops)
    # add passengers to the graph
    index_p = 0
    for x in range(num_passengers):       
        passenger = Passenger(index_p, passengers[x,1], passengers[x,2], passengers[x,3],passengers[x,4])
        graph.add_passenger(passenger)
        index_p += 1
    # add stops to the graph
    index_s = 0
    for x in range(num_stops):
        stop = Stop(index_s, stops[x,1], stops[x,2])
        graph.add_stop(stop)
        index_s += 1
    # compute global distance and find near stops for passengers
    graph.cal_distance()
    graph.get_near_stops()
        
    return graph

def initial_route(graph):
# initial routes
    for passenger in graph.passengers:
        for board_stop in passenger.home_near_stops:
            for alight_stop in passenger.company_near_stops:
                route = Route([board_stop], [alight_stop], [passenger])
                passenger.home_stop = board_stop
                passenger.company_stop = alight_stop                
                graph.add_route(route)

"""
Compute Saving List

1 - define a saving list
2 - for each passenger
        if 2 passengers are not the same, compute the delta profit
            delta = fare*(d(j)+d(i to j)+d(j to i)-d(i)) + cost*(d(i)- (d(i to j)+d(j to i))) 
4 - append the list with passenger combination (i,j) and its saving value
5 - sort the list by saving value
6 - define route i is outter, route j is inner
7 - the delta distance is the distance for i merge to j

""" 
def cal_save(passenger1, passenger2, graph):
    
    delta_profit_list = []
    routes1 = []
    routes2 = []
    #find the routes where contains passenger i 
    for i in graph.routes:
        if passenger1 in i.passengers:
           routes1.append(i)
    #find the routes where contains passenger j 
    for j in graph.routes:
        if passenger2 in j.passengers:
           routes2.append(j)

    for i in routes1:
        for j in routes2:         
            d1 = i.board_stop[-1].distance_stop_stop[j.board_stop[0]] #d(i to j)
            d2 = j.board_stop[0].distance_stop_stop[j.alight_stop[-1]] #d(j)
            d3 = j.alight_stop[-1].distance_stop_stop[i.alight_stop[0]] #d(j to i)
            d4 = i.board_stop[0].distance_stop_stop[i.alight_stop[-1]] #d(i)
            delta_profit = fare*(d2 + d1 + d3 - d4) + cost*(d4- (d1 + d3)) #profit_save
            
            delta_profit_list.append([[passenger1, passenger2],delta_profit])
    
    return delta_profit_list

def save_list(graph):

    saving_list = []
    for i in graph.passengers:
        for j in graph.passengers:
            if i != j: 
                x = cal_save(i, j, graph)
                saving_list.extend(x)
    #save for profit
    saving_list = [i for i in saving_list if i[1] > 0]
    saving_list.sort(key=lambda x: x[1], reverse=True)
    
    return saving_list

"""
Check Feasible Merge

# define functions:
1 - define functions to compute board distance, alight distance, route distance for 1 route
2 - define functions to compute total distance, arrival distance, detour ration when merge 2 routes

# check if passenger i and passenger j can be merged:
1 - set outter routes list and inner routes list
2 - for each route in routes list
3 - if a route's first board stop is j's board stop, and the last alight stop is j's alight stop 
    (which is inner)  
4 - set it as the inner route 
5 - if a route's last board stop is i's board stop, and the first alight stop is i's alight stop 
    (which is outter) 
6 - set it as the outter route
7 - if the 2 routes satisfied the total distance, arrival distance and detour ratio constrain
    the 2 routes could be merged, otherwise not 

"""
def merge_dist(route1,route2):

    original_dist = route1.cal_board_dist() + route1.cal_alight_dist() + route2.cal_distance() 
    connect_dist = route1.board_stop[-1].distance_stop_stop[route2.board_stop[0]] + route2.alight_stop[-1].distance_stop_stop[route1.alight_stop[0]]
    merge_dist = original_dist + connect_dist
    
    return merge_dist

def arrive_dist(route1,route2):
 
    arrive_dist = route1.cal_alight_dist() + route2.cal_alight_dist() + route2.alight_stop[-1].distance_stop_stop[route1.alight_stop[0]]
    
    return arrive_dist

def detour_ratio(route1,route2):
 
    detour = merge_dist(route1,route2) / route1.board_stop[0].distance_stop_stop[route1.alight_stop[-1]]
    
    return detour
    
def feasible_merge(passenger1, passenger2, graph):
    
    outer = 0 
    inner = 0      
    
    for route in graph.routes: 
            
        if (passenger1 in route.passengers) and (route.board_stop[-1] in passenger1.home_near_stops) and (route.alight_stop[0] in passenger1.company_near_stops): 
            outer = route 
            
        elif (passenger2 in route.passengers) and (route.board_stop[0] in passenger2.home_near_stops) and (route.alight_stop[-1] in passenger2.company_near_stops): 
            inner = route 
        
    if outer in graph.routes and inner in graph.routes: 
        if (outer.demand + inner.demand <= capacity):
            if (merge_dist(outer,inner) <= max_distance):
                if (arrive_dist(outer,inner) <= min_arrival):
                    if (detour_ratio(outer,inner) <= max_detour):
                        return True 
"""
Merge Routes

1 - set inner and outter route for passenger i and j
2  -set a mix route list
3 - mix's board stop is outter + inner
4 - mix's alight stop is inner + outter
5 - mix's demand is outter + inner
6 - delete outter and inner in routes
7 - append mix to routes

"""
def merge_routes(passenger1, passenger2, graph):
    
    outer = 0 
    inner = 0    
#    if (route1 in routes) and (route2 in routes):        
    for route in graph.routes:
        if (passenger1 in route.passengers) and (route.board_stop[-1] in passenger1.home_near_stops) and (route.alight_stop[0] in passenger1.company_near_stops): 
            outer = route 
                
            passenger1.home_stop = route.board_stop[-1]
            passenger1.home_near_stops = [route.board_stop[-1]]
            passenger1.company_stop = route.alight_stop[0]
            passenger1.company_near_stops = [route.alight_stop[0]]

        elif (passenger2 in route.passengers) and (route.board_stop[0] in passenger2.home_near_stops) and (route.alight_stop[-1] in passenger2.company_near_stops): 
            inner = route
            
            passenger2.home_stop = route.board_stop[0]
            passenger2.home_near_stops = [route.board_stop[0]]
            passenger2.company_stop = route.alight_stop[-1]
            passenger2.company_near_stops = [route.alight_stop[-1]]
    
    routes1 = []
    routes2 = []
    #find the routes where contains passenger i 
    for route1 in graph.routes:
        if passenger1 in route1.passengers:
           routes1.append(route1)
    #find the routes where contains passenger j 
    for route2 in graph.routes:
        if passenger2 in route2.passengers:
           routes2.append(route2)
           
    merge = Route(outer.board_stop + inner.board_stop, inner.alight_stop + outer.alight_stop, outer.passengers + inner.passengers)
    merge.demand = outer.demand + inner.demand

    for route in routes1:
        graph.routes.remove(route)
    for route in routes2:
        graph.routes.remove(route)
        
    graph.add_route(merge)    

"""
Plot Solution

1 - set basemap,region,size
2 - define functions to plot od, odlines, stops and routes
3 - plot them in the map

"""
def plot_solution(graph):

#def plot_od_point(graph):    
    for passenger in graph.passengers:
        o_x, o_y = map(passenger.home_x, passenger.home_y)
        d_x, d_y = map(passenger.company_x, passenger.company_y)
        
        map.scatter(o_x, o_y, s=5, c='g', marker = 'o', alpha = 0.3)
        map.scatter(d_x, d_y, s=5, c='r', marker = '^', alpha = 0.3)

#def plot_od_line(graph):
    for passenger in graph.passengers:
        od_x, od_y = map(np.append(passenger.home_x,passenger.company_x), np.append(passenger.home_y,passenger.company_y))
        map.plot(od_x, od_y, c='k',alpha = 0.04,linewidth = 1)

#def plot_stop_point(graph):  
#    for stop in graph.stops:
#        stop_x, stop_y = map(stop.x, stop.y)
#        map.scatter(stop_x, stop_y, s=5, c='k', marker = '+', alpha = 0.01)
         
#def plot_passenger_stop(graph):    
    for route in graph.routes[0:max_vehicle]:
        for passenger in route.passengers:
            hs_x, hs_y = map(np.append(passenger.home_x, passenger.home_stop.x), np.append(passenger.home_y, passenger.home_stop.y))
            map.plot(hs_x, hs_y, c='k',alpha = 1,linewidth = 1)
            
            cs_x, cs_y = map(np.append(passenger.company_x, passenger.company_stop.x), np.append(passenger.company_y, passenger.company_stop.y))
            map.plot(cs_x, cs_y, c='k',alpha = 1,linewidth = 1)

#def plot_route(graph):   
    for route in graph.routes[0:max_vehicle]:
        
        lon1,lat1,lon2,lat2 = [],[],[],[]
        color = random.choice('rgbcmky')
        
        for stop in route.board_stop:
            lon1.append(stop.x)
            lat1.append(stop.y)
            
        for stop in route.alight_stop:
            lon2.append(stop.x)
            lat2.append(stop.y)
              
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

def print_solution(graph):
    
    print('='*32)
    print('total passengers:', graph.num_passengers)
    print('total stops:', graph.num_stops) 
    time_end=time.time()
    print('time cost:',time_end-time_start,'s')
    
    id = 0
    print('='*32)
    for route in graph.routes[0:max_vehicle]:
        print('{}={}'.format('route_index',id))
        print('{}={}'.format('route_demand',route.demand))
        print('{}={}'.format('route_profit',route.profit))
        print('='*32)
        id += 1
    
"""
Main Process

1 - create initial routes
    routes = [ [board_stop,...],[alight_stop,...],demand,[passenger,...] ]
2 - compute list of savings
3 - for each save combination in the list
        check each passenger i,j combination see if they are able to merge
        if feasible: merge
4 - return solution list
5 - sort the solution list by profit of each route in solution
6 - find the max number of routes according to the number of avaliable vehicles

"""
time_start=time.time()
capacity = 40  
max_distance = 20
min_walk_dist = 0.2
min_arrival = 3
max_vehicle = 5
max_detour = 1.5
cost = 9
fare = 2
lng1,lat1 = 106.607665,26.49229 # lng1,lat1: lower left conner
lng2,lat2 = 106.819152,26.708405 # lng2,lat2: upper right conner
p_file_name = "passengers_file_2954.csv"
s_file_name = "stops_file.csv"

graph = read(p_file_name, s_file_name)
initial_route(graph)
save_list = save_list(graph)
for saves in save_list:
    i, j = saves[0]
    if (feasible_merge(i, j, graph)):
        merge_routes(i, j, graph)
graph.sort_routes()

map = Basemap(projection='mill',
            llcrnrlat = lat1-0.01, # left corner latitude
            llcrnrlon = lng1-0.01, # left corner longitude
            urcrnrlat = lat2+0.01, # right corner latitude
            urcrnrlon = lng2+0.01, # right corner longitude
            resolution='l', 
            area_thresh=1000)
fig = plt.figure(figsize=(20,20))
ax = plt.gca()
plot_solution(graph)
print_solution(graph)


