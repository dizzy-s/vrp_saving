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
        self.top_profit = 0
        self.total_profit = 0
        
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
        total_cost = fixed_cost + cost * self.cal_distance()
        
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
        self.top_profit = 0
        self.top_demand = 0
        self.total_profit = 0
        
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
        
        self.top_profit = 0
        for route1 in self.routes[0:max_vehicle]:
            self.top_profit += route1.profit
        
        self.top_demand = 0
        for route1 in self.routes[0:max_vehicle]:
            self.top_demand += route1.demand
        
        self.total_profit = 0
        for route2 in self.routes:
            self.total_profit += route2.profit
                
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
Compute Saving

assuming passenger i is in the outer route, passenger j is in the inner route
the delta profit is the distance cost saving for route i merge to j

# compute saving for passenger i and j:
1 - find each route in graph where contains passenger i
    append it in route list 1
2 - find each route in graph where contains passenger j
    append it in route list 2
3 - for each route in route list 1:
        for each route in route list 2:
            original profit = fare * d(i) - cost * d(i) + fare * d(j) - cost * d(j)
            merge profit = fare * d(i) + fare * d(j) - cost * ( d(i to j) + d(j) + d(j to i) )            
            delta profit = d(i) - ( d(i to j) + d(j to i) )
            append it in delta profit list for all possible merging for i and j
            
# compute saving list for graph:
1 - define a saving list
2 - for each passenger i:
        for each passenger j:
            if 2 passengers are not the same, compute the delta profit for i merging to j:
                compute saving for passenger i and j
4 - extend the list with passenger combination (i,j) and its saving value
5 - sort the list by saving value, keep only positive combinations

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
#            d2 = j.board_stop[0].distance_stop_stop[j.alight_stop[-1]] #d(j)
            d3 = j.alight_stop[-1].distance_stop_stop[i.alight_stop[0]] #d(j to i)
            d4 = i.board_stop[0].distance_stop_stop[i.alight_stop[-1]] #d(i)
#            delta_profit = fare*(d2 + d1 + d3 - d4) + cost*(d4- (d1 + d3)) #profit_save
            delta_profit = d4- (d1 + d3) #profit_save
            
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
Feasible Merge

# define functions:
1 - define functions to compute board distance, alight distance, route distance for 1 route
2 - define functions to compute total distance, arrival distance, detour ration when merge 2 routes

# check if passenger i and passenger j can be merged:
1 - initial outer route and inner route
2 - for each route in routes list:
3 - if passenger i in the route, a route's last board stop is i's board stop, and the first alight stop is i's alight stop 
        set it as the outer route
4 - if passenger j in the route, a route's first board stop is j's board stop, and the last alight stop is j's alight stop 
        set it as the inner route 
5 - if the 2 routes satisfied the capacity, total distance, arrival distance and detour ratio constraints
        the 2 routes could be merged, otherwise not 

# merge routes
1 - initial a merge route instance
2 - merge's board stop is outer + inner
3 - merge's alight stop is inner + outer
4 - merge's demand is outer + inner
5 - delete all routes containing i and all routes containing j in route list
6 - add merge to graph
 
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
def plot_origin(graph):

#def plot_od_point(graph):    
    for passenger in graph.passengers:
        o_x, o_y = map1(passenger.home_x, passenger.home_y)
        d_x, d_y = map1(passenger.company_x, passenger.company_y)
        
        map1.scatter(o_x, o_y, s=5, c='g', marker = 'o', alpha = 0.1)
        map1.scatter(d_x, d_y, s=5, c='r', marker = '^', alpha = 0.1)

#def plot_od_line(graph):
    for passenger in graph.passengers:
        od_x, od_y = map1(np.append(passenger.home_x,passenger.company_x), np.append(passenger.home_y,passenger.company_y))
        map1.plot(od_x, od_y, c='k',alpha = 0.04,linewidth = 1)

#def plot_stop_point(graph):  
#    for stop in graph.stops:
#        stop_x, stop_y = map1(stop.x, stop.y)
#        map1.scatter(stop_x, stop_y, s=5, c='k', marker = '+', alpha = 0.01)

def plot_solution(graph,number):         
#def plot_passenger_stop(graph):    
    for route in graph.routes[0:number]:
        for passenger in route.passengers:
            hs_x, hs_y = map1(np.append(passenger.home_x, passenger.home_stop.x), np.append(passenger.home_y, passenger.home_stop.y))
            map1.plot(hs_x, hs_y, c='k',alpha = 0.8,linewidth = 0.5)
            
            cs_x, cs_y = map1(np.append(passenger.company_x, passenger.company_stop.x), np.append(passenger.company_y, passenger.company_stop.y))
            map1.plot(cs_x, cs_y, c='k',alpha = 0.8,linewidth = 0.5)

#def plot_route(graph):   
    for route in graph.routes[0:number]:
        
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
        
        x1, y1 = map1(lon1, lat1)
        x2, y2 = map1(lon2, lat2)
        x3, y3 = map1(np.append(lon1,lon2), np.append(lat1,lat2))
        
        map1.scatter(x1, y1, s=50, c=color, alpha = 0.2, marker = 'o')
        map1.scatter(x2, y2, s=50, c=color, alpha = 0.2, marker = '^')   
        map1.plot(x3, y3, c=color)        

def print_solution(graph,number):
    
    print('='*32)
    print('total passengers:', graph.num_passengers)
    print('total stops:', graph.num_stops)
    print('top_demand:', graph.top_demand)
    print('top profit:', graph.top_profit)
    time_end=time.time()
    print('{}: {:.4} {}'.format('time cost',(time_end-time_start)/60, 'min' ))
    
    id = 1
    print('='*32)
    for route in graph.routes[0:number]:
        print('{}={}'.format('route_index',id))
        print('{}={}'.format('route_demand',route.demand))
        print('{}={:.4}'.format('route_profit',route.profit))
        print('='*32)
        id += 1

"""
Meta-heuristics
            
# exchange position operator (exchange board and alight seperately)
1 - randomly choose 1 route
2 - randomly choose 2 passengers in route:        
3 - exchange the position of their board stop or alight stop
4 - if feasible:
        if new profit > old profit:
            remove the old route
            add the exchange route

# remove operator
1 - randomly choose 1 route
2 - randomly choose 1 passenger in route
3 - remove the passenger from that route
4 - the passenger become 1 independent route
5 - if feasible:
        if new profit > old profit:
            remove the old route
            add the new 2 routes

# remove and insert operator
1 - randomly choose 2 routes
2 - randomly choose 1 passenger in route 1:        
3 - find the position for board and alight stops and add into route 2
4 - if feasible:
        if new profit(route 2) > old profit(route 2):
            add the passenger to route 2
            remove the passenger from route 1
            
"""
def dist(stop1,stop2):
    dist = get_distance(stop1.x,stop1.y,stop2.x,stop2.y)
    return dist

def detour(route):
    if route.board_stop[0].distance_stop_stop[route.alight_stop[-1]] != 0:
        route_detour = route.distance / route.board_stop[0].distance_stop_stop[route.alight_stop[-1]]
    else: 
        route_detour = route.distance / 0.0001    
    return route_detour

def feasible(route):
    if (route.demand <= capacity):
        if (route.distance <= max_distance):
            if (route.alight_dist <= min_arrival):
                if (detour(route) <= max_detour):
                    return True
                else:
                    return False

def exchange_board(graph):
        
    route = random.choice(graph.routes[0:max_vehicle])
    
    if len(route.passengers) >= 2:            
        passenger1, passenger2 = random.sample(route.passengers, 2)
        
        exchange_board = 0
        index1 = route.board_stop.index(passenger1.home_stop)
        index2 = route.board_stop.index(passenger2.home_stop)
        board_list = route.board_stop.copy()
        board_list[index1],board_list[index2] = board_list[index2],board_list[index1]
        alight_list = route.alight_stop.copy()
        passenger_list = route.passengers.copy()
        exchange_board = Route(board_list, alight_list, passenger_list)  
        exchange_board.demand = len(passenger_list)
        exchange_board.cal_distance()
        exchange_board.cal_profit()
            
        if feasible(exchange_board):
            if exchange_board.profit > route.profit:
                graph.routes.remove(route)
                graph.add_route(exchange_board)     

def exchange_alight(graph):
        
    route = random.choice(graph.routes[0:max_vehicle])
    
    if len(route.passengers) >= 2:            
        passenger1, passenger2 = random.sample(route.passengers, 2)
        
        exchange_alight = 0
        index1 = route.alight_stop.index(passenger1.company_stop)
        index2 = route.alight_stop.index(passenger2.company_stop)
        board_list = route.board_stop.copy()
        alight_list = route.alight_stop.copy()
        alight_list[index1],alight_list[index2] = alight_list[index2],alight_list[index1]
        passenger_list = route.passengers.copy()
        exchange_alight = Route(board_list, alight_list, passenger_list)  
        exchange_alight.demand = len(passenger_list)
        exchange_alight.cal_distance()
        exchange_alight.cal_profit()
            
        if feasible(exchange_alight):
            if exchange_alight.profit > route.profit:
                graph.routes.remove(route)
                graph.add_route(exchange_alight)     

def remove_left(graph):
    
    route = random.choice(graph.routes[0:max_vehicle])
    
    if len(route.passengers) >= 2:            
        passenger = random.choice(route.passengers)
        
        remove = 0
        board_list = route.board_stop.copy()
        board_list.remove(passenger.home_stop)
        alight_list = route.alight_stop.copy()
        alight_list.remove(passenger.company_stop)
        passenger_list = route.passengers.copy()
        passenger_list.remove(passenger)
        remove = Route(board_list, alight_list, passenger_list)  
        remove.demand = len(passenger_list)
        remove.cal_distance()
        remove.cal_profit()

        left = 0
        board_list = [passenger.home_stop]
        alight_list = [passenger.company_stop]
        passenger_list = [passenger]
        left = Route(board_list, alight_list, passenger_list)  
        left.demand = len(passenger_list)
        left.cal_distance()
        left.cal_profit()

        if feasible(remove):
            if remove.profit > route.profit:
                graph.routes.remove(route)
                graph.add_route(remove)
                graph.add_route(left)

def remove_insert(graph):  
    # route1: to be removed; route2: to be inserted       
    route1, route2, passenger = 0, 0, 0
    route2 = random.choice(graph.routes[0:max_vehicle])
    graph.routes_temp = graph.routes.copy()
    graph.routes_temp.remove(route2)
    route1 = random.choice(graph.routes_temp)
    passenger = random.choice(route1.passengers)
        
    board_list = []
    board_list = route2.board_stop.copy()
    board_list.insert(0,depot)
    board_list.append(route2.alight_stop[0])
    
    alight_list = []
    alight_list = route2.alight_stop.copy()
    alight_list.insert(0,route2.board_stop[-1])
    alight_list.append(depot)
    
    dist_board = []
    for i in range(1,len(board_list)):
       dist_board.append(dist(board_list[i-1],passenger.home_stop) + dist(passenger.home_stop,board_list[i]) - dist(board_list[i-1],board_list[i]))
    index = dist_board.index(min(dist_board))
    board_list.pop(0)
    board_list.pop(-1)
    board_list.insert(index,passenger.home_stop)
    
    dist_alight = []
    for i in range(1,len(alight_list)):
       dist_alight.append(dist(alight_list[i-1],passenger.company_stop) + dist(passenger.company_stop,alight_list[i]) - dist(alight_list[i-1],alight_list[i]))
    dist_alight.index(min(dist_alight))
    alight_list.pop(0)
    alight_list.pop(-1)
    alight_list.insert(index,passenger.company_stop)
    
    insert = 0
    passenger_list = route2.passengers.copy()
    passenger_list.append(passenger)
    insert = Route(board_list, alight_list, passenger_list)    
    insert.demand = len(passenger_list)
    insert.cal_distance()
    insert.cal_profit()

    remove = 0        
    if len(route1.passengers) > 1:
    # if the remove route has more than 1 passengers
        board_list = route1.board_stop.copy()
        board_list.remove(passenger.home_stop)
        alight_list = route1.alight_stop.copy()
        alight_list.remove(passenger.company_stop)
        passenger_list = route1.passengers.copy()
        passenger_list.remove(passenger)         
        remove = Route(board_list, alight_list, passenger_list)  
        remove.demand = len(passenger_list)
        remove.cal_distance()
        remove.cal_profit()
           
        if feasible(insert):
            if (route1 not in graph.routes[0:max_vehicle] and insert.profit > route2.profit) or (route1 in graph.routes[0:max_vehicle] and insert.profit + remove.profit > route2.profit + route1.profit):
                graph.routes.remove(route1)
                graph.routes.remove(route2)                        
                graph.add_route(insert)
                graph.add_route(remove)
                
    else:
    # if the remove route has only 1 passenger
        if feasible(insert):
            if (route1 not in graph.routes[0:max_vehicle] and insert.profit > route2.profit) or (route1 in graph.routes[0:max_vehicle] and insert.profit + remove.profit > route2.profit + route1.profit):
                    graph.routes.remove(route1)
                    graph.routes.remove(route2)
                    graph.add_route(insert)
                
"""
Saving-heuristics Process

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

"""Guiyang CB Input"""
#capacity = 40  
#max_distance = 30
#min_walk_dist = 0
#min_arrival = 5
#max_vehicle = 5
#max_detour = 1.5
#fixed_cost = 100
#cost = 2
#fare = 1
#lng1,lat1 = 106.606705,26.499468 # lng1,lat1: lower left conner
#lng2,lat2 = 106.794159,26.702689 # lng2,lat2: upper right conner
#p_file_name = "passengers_file_2954.csv"
#s_file_name = "stops_file.csv"

"""Beijing HUB Input"""
capacity = 10  
max_distance = 30
min_walk_dist = 0
min_arrival = 5
max_vehicle = 10
max_detour = 1.5
fixed_cost = 20
cost = 0.5
fare = 1.2
lng1,lat1 = 116.177319,39.775079 # lng1,lat1: lower left conner
lng2,lat2 = 116.671703,40.150846 # lng2,lat2: upper right conner
p_file_name = "passengers_file_bjhlg_30min_1day.csv"
s_file_name = "stops_file_bj.csv"

graph = read(p_file_name, s_file_name)
initial_route(graph)
save_list = save_list(graph)
for saves in save_list:
    i, j = saves[0]
    feasible_merge(i, j, graph)
graph.sort_routes()

"""
Plot Initial Solution
"""
map1 = Basemap(projection='mill',
            llcrnrlat = lat1-0.01, # left corner latitude
            llcrnrlon = lng1-0.01, # left corner longitude
            urcrnrlat = lat2+0.01, # right corner latitude
            urcrnrlon = lng2+0.01, # right corner longitude
            resolution='l')
fig = plt.figure(figsize=(20,20))
#plot_origin(graph)
plot_solution(graph,max_vehicle)
print_solution(graph,max_vehicle)

"""
Meta-heuristics Process

1 - add a dummy depot to the graph
2 - initial profit list
3 - set max_iter
4 - sequencially execute the 4 operators
5 - iter + 1
5 - sort the route list by profit
6 - end if it reaches the max_iter

"""
depot = Stop(99999,0,0)
graph.add_stop(depot)

top_profit_list = [graph.top_profit]    
top_demand_list = [graph.top_demand]

max_iter = 100000

iter = 1
while iter <= max_iter:

    exchange_board(graph)
    exchange_alight(graph)
    remove_left(graph)
    remove_insert(graph)
    
#    operator_num = random.choice([1,2,3,4])   
#    if operator_num == 1:
#        remove_insert(graph)
#    if operator_num == 2:
#        exchange_board(graph)
#    if operator_num == 3:
#        exchange_alight(graph)
#    if operator_num == 4:
#        remove_left(graph)
    
    graph.sort_routes()
    top_profit_list.append(graph.top_profit)
    top_demand_list.append(graph.top_demand)
    
    iter += 1

"""
Plot Final Solution
"""
map1 = Basemap(projection='mill',
            llcrnrlat = lat1-0.01, # left corner latitude
            llcrnrlon = lng1-0.01, # left corner longitude
            urcrnrlat = lat2+0.01, # right corner latitude
            urcrnrlon = lng2+0.01, # right corner longitude
            resolution='l')
fig = plt.figure(figsize=(20,20))
plot_origin(graph)
plot_solution(graph,max_vehicle)
print_solution(graph,max_vehicle)

fig = plt.figure(figsize=(10,10))
plt.plot(top_demand_list)
plt.plot(top_profit_list)

