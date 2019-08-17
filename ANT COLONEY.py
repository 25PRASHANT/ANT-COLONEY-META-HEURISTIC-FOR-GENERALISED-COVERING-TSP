# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 19:58:02 2019

@author: PRASHANT
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:56:06 2019

@author: PRASHANT
"""

from gurobipy import*
import os
import xlrd
import numpy as np
from numpy import inf
from scipy import spatial
import numpy
from sklearn.metrics.pairwise import euclidean_distances
import math
iteration = 100
n_ants =16
n_facility = 16         ####### 1 PLUS ACTUAL NO OF FACILITES
n_cust=35
e=0.5
alpha=2
break_limit=20
beta=2
minimum_demand=35
nc=6            ###ALSO CORRECT LINE 89
facility_cordinate={}
cust_cordinate={}

demand = np.ones((n_cust))

book = xlrd.open_workbook(os.path.join("1.xlsx"))
sh = book.sheet_by_name("Sheet1")
i = 1
l=0
for i in range(1,n_facility+1):  
    sp = sh.cell_value(i,1)
    sp2 = sh.cell_value(i,2)
    sp1=(sp,sp2)
    facility_cordinate[l]=(sp1)
    l=l+1
j=0
for i in range(n_facility+1,n_cust+1+n_facility):  
    sp = sh.cell_value(i,1)
    sp2 = sh.cell_value(i,2)
    sp1=(sp,sp2)
    cust_cordinate[j]=sp1  
    j=j+1
    
def calculate_dist(x1, x2):
    eudistance = spatial.distance.euclidean(x1, x2)
    return(eudistance)

f_dist=[]
     
for i in facility_cordinate:
    facility_dist=[]
#    a=facility_cordinate[i]
    for j in facility_cordinate:
        facility_dist.append(calculate_dist(facility_cordinate[i],facility_cordinate[j]))
    f_dist.append(facility_dist)
fac_dist=np.array(f_dist)    
customer_dist={}
for i in facility_cordinate:
    if i!=0:
        for j in cust_cordinate:
            customer_dist[i,j]=calculate_dist(facility_cordinate[i],cust_cordinate[j])

abc={}
for i in range(1,n_facility+1):
    j = 3
    xyz=[]
    while True:
        try:
            
            sp = sh.cell_value(i,j)
            xyz.append(sp)
            
            j = j + 1
            
        except IndexError:
            break
    abc[i-1]=xyz
    
final_aij=[]
for i in range(n_facility):
    a_ij=[]
    for j in range(n_facility,n_facility+n_cust):
        if j in abc[i]:
            a_ij.append(1)
        else:
            a_ij.append(0)
    final_aij.append(a_ij)
cust_dist=np.array(final_aij)
#
###
zx=[]

for i in range(n_cust):
    zx.append(i)
zx.sort()
#################################################################
################################################################
demand = demand[:,np.newaxis]
pheromne = 0.15*np.ones((n_ants,n_facility))

route = np.ones((n_ants,n_facility))

#newArray=np.array([])

no_of_cust_covered = np.zeros((1,n_facility))
dem_sat_array=np.zeros((1,n_facility))


for num1 in range(n_facility):
    s=0
    dem_sat=0
    for num2 in range(n_cust):
        if cust_dist[num1,num2]==1:
                    s+=1
                    dem_sat+=demand[num2]
    no_of_cust_covered[0,num1]=s
    dem_sat_array[0,num1]=dem_sat

factor=1/fac_dist    
factor[factor==inf]=0
visibility=factor*dem_sat_array

overall_dist_min_cost=10000000
cost_matrix=np.zeros((iteration,1))


for ite in range(iteration):             #iteration
    

    route = np.ones((n_ants,n_facility))#####*******
    for i in range(n_ants):              # no of ants
        
        temp_visibility = np.array(visibility)
        demand_satisfied=0
        unsatisfied_cust=[]
        satisfied_cust=[]
        temp_no_of_cust_covered=np.array(no_of_cust_covered)
        temp_dem_sat_array=np.array(dem_sat_array)
        
        temp_cust_distance=np.array(cust_dist)


        for j in range(n_facility-1):
            fac=zx[:]
            
            if (demand_satisfied<minimum_demand):
                if j>0:
                    bahubali=[]
                    for n1 in range(n_facility):
                        s=0
                        dem_sat=0
                        for n2 in range(n_cust):
                            if temp_cust_distance[n1,n2]==1:
                                s+=1
                                dem_sat+=demand[n2]
                        temp_no_of_cust_covered[0,n1]=s
                        temp_dem_sat_array[0,n1]=dem_sat
                        
                    for a1 in range(n_facility):
                        if temp_dem_sat_array[0,a1]==0:
##                        if temp_no_of_cust_covered[0,a1]==0:
                           
                            bahubali.append(a1)
#                            temp_visibility[:,a1] = 0


#                    no_of_cust_covered[facility-1,0]=1
                    temp_visibility=factor*temp_dem_sat_array
                    for b1 in bahubali:
                        temp_visibility[:,b1]=0
#                temp_visibility[temp_visibility==inf]=0
####                unsatisfied_cust=[]
                demand_satisfied=0
                combine_feature = np.zeros(n_facility)
                cum_prob = np.zeros(n_facility)
                cur_loc = int(route[i,j]-1)
                temp_visibility[:,cur_loc] = 0
                p_feature = np.power(pheromne[cur_loc,:],beta)
                v_feature = np.power(temp_visibility[cur_loc,:],alpha)
                p_feature = p_feature[:,np.newaxis]
                v_feature = v_feature[:,np.newaxis]
                combine_feature = np.multiply(p_feature,v_feature)
                total = np.sum(combine_feature)
                probs = combine_feature/total
                cum_prob = np.cumsum(probs)
                r = np.random.random_sample()
            
                facility = np.nonzero(cum_prob>r)[0][0]+1
                route[i,j+1] = facility
#                z1=0
                for k in range(n_cust):
                    if (temp_cust_distance[facility-1,k]==1):
                       satisfied_cust.append(k)
                       
                for g1 in satisfied_cust:
                    fac.remove(g1)
                unsatisfied_cust=fac

                for b in satisfied_cust:
                    demand_satisfied+=np.sum(demand[b,0])
##                    
##                cust_dist[facility-1,:]=0
                for a in satisfied_cust:
                    temp_cust_distance[:,a]=0
            else:
                break
    route_opt = np.array(route)               #intializing optimal route
    
    dist_cost = np.zeros((n_ants,1))             #intializing total_distance_of_tour with zero 
    
    for i in range(n_ants):
        
        abcd = 0
        for j in range(n_facility-1):
            
            abcd = abcd + fac_dist[int(route_opt[i,j])-1,int(route_opt[i,j+1])-1]   #calcualting total tour distance
        
        dist_cost[i]=abcd                      #storing distance of tour for 'i'th ant at location 'i' 
       
    dist_min_loc = np.argmin(dist_cost)             #finding location of minimum of dist_cost
    dist_min_cost = dist_cost[dist_min_loc]         #finging min of dist_cost

    cost_matrix[ite]=dist_min_cost               ##BREAKING CRITERIA
    if ite>break_limit:
        out=0
        for v in range(ite,ite-break_limit,-1):
            if cost_matrix[v]==cost_matrix[v-1]:
                out+=1
        if out==break_limit:
            break
    
    
#    print(dist_min_cost)
 #   print(route_opt)
    
    
    if dist_min_cost < overall_dist_min_cost:
        overall_dist_min_cost=dist_min_cost
        overall_best_route=route[dist_min_loc,:]
    
    best_route = route[dist_min_loc,:]               #intializing current traversed as best route
    
#    print('best path :',best_route)
    
    
    pheromne = (1-e)*pheromne
    for c in range(n_ants):
        for v in range(n_facility-1):
            dt = 1/dist_cost[c]
            pheromne[int(route_opt[c,v])-1,int(route_opt[c,v+1])-1] = pheromne[int(route_opt[c,v])-1,int(route_opt[c,v+1])-1] + dt
            

print('route of all the ants at the end :')
print(route_opt)
print()
print('best path :',overall_best_route)
#print('cost of the best path',int(dist_min_cost[0]) + fac_dist[int(best_route[-2])-1,0])  
print('cost of the best path',overall_dist_min_cost)        
    
    