import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import matplotlib.pyplot as plt
import scipy.misc
import os
import scipy.stats as ss
import math
#matplotlib inline
import os, sys
import datetime
## SUMO控制协议
import traci
import utils as util
import traci.constants as tc
import arrivalGen as ag
# if 'SUMO_HOME' in os.environ:
# The path of SUMO-tools to get the traci library

## 主要是导入一些常用的shell命令
## for linux
os.environ['SUMO_HOME'] = "/myfile/sumo-0.32.0/"
sys.path.append("/myfile/sumo-0.32.0/tools")
## for windows
# sys.path.append(os.path.join('E:/sumo-0.32.0/', 'tools'))
# sys.path.append('E:/sumo-0.32.0/')

# Environment Model
# sumoBinary = "/usr/local/bin/sumo"
sumoBinary = "sumo"      ## 控制命令，可以在sumo和sumo-gui之间切换
road_path = "../data/map1.sumocfg"                 # the road path
sumoCmd = [sumoBinary, "-c", road_path, "--start"]  # The path to the sumo.cfg file
class traffic_env():
    def __init__(self, interval = 2):
        self.interval = interval
        self.sumoCmd = sumoCmd
        self.sumoCmd_gui = ["sumo-gui", "-c", road_path, "--start"]
        traci.start(["sumo", "-c", road_path, "--start"])
        self.tls = traci.trafficlights.getIDList()
        self.actionsMap = util.makemap(self.tls)
        self.n_act = len(self.actionsMap)
        ID_edge = traci.edge.getIDList()
        self.ID_edge = [v for v in ID_edge if 'edge' in v]
        state = self.get_state()
        self.n_sta = len(state)
        self.entrance = []
        self.exits = []
        self.unreachable = {}
        for i, iedge in enumerate(self.ID_edge):
            if i <=3 :
                self.exits.append(iedge)
            else:
                self.entrance.append(iedge)
        for i in range(len(self.entrance)):
            self.unreachable[self.entrance[i]] = self.exits[i]
        self.end()
    def reset(self, cmd):
        traci.start(cmd)
    def start(self):
        state = self.get_state()
        self.lwait_time = self.get_waittime()
        return state
    def end(self):
        traci.close()
    def step(self, action):
        lightsPhase = self.actionsMap[action]
        for light, index in zip(self.tls, range(len(self.tls))):
            traci.trafficlights.setPhase(light, lightsPhase[index])
        for i in range(self.interval):
            traci.simulationStep()
        state_ = self.get_state()
        reward = self.get_reward()
        return reward, state_
    def generateTrips(self, whole_step = 1000):
        flow = [0.2, 0.2, 0.2, 0.2]
        tripsList = []
        for i in range(whole_step):
            for j in range(len(self.entrance)):
                enEdge = self.entrance[j]
                f = flow[j]
                p = random.uniform(0,1)
                if p > f:
                    continue
                p = int(random.uniform(0,len(self.exits)))
                des = self.exits[p]
                while des == self.unreachable[enEdge]:
                    p = int(random.uniform(0,len(self.exits)))
                    des = self.exits[p]
                tripsList.append([i, enEdge, des])
        trip_path = '../data/map1.trips.xml'
        ag.writeTripsXml(tripsList, trip_path)
        net_path = '../data/map1.net.xml'
        rou_path = '../data/map1.rou.xml'
        os.system('duarouter -n ' + net_path +  ' -t ' + trip_path + ' -o ' + rou_path)
    def get_state(self):
        state_ = []
        for iedge in self.ID_edge:
            haltingNum = traci.edge.getLastStepHaltingNumber(iedge)
            meanSpeed = traci.edge.getLastStepMeanSpeed(iedge)
            if meanSpeed == 40:
                meanSpeed = 0
            state_.append(haltingNum)
            state_.append(meanSpeed)
        return state_
    def get_reward(self):
        self.cwait_time = self.get_waittime()
        reward = self.lwait_time - self.cwait_time
        reward -= self.get_queue()
        self.lwait_time = self.cwait_time
        return  reward
    def get_waittime(self):
        wait_time_map = {}
        for veh_id in traci.vehicle.getIDList():
            wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        wait_temp = dict(wait_time_map)
        wait_sum_time = sum(wait_temp[x] for x in wait_temp)
        return wait_sum_time
    def get_queue(self):
        queue_len = []
        for iedge in self.entrance:
            length = traci.edge.getLastStepVehicleNumber(iedge)
            queue_len.append(length)
        return sum(queue_len)











