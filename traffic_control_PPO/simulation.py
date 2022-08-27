import traci
import traci.constants as tc
import time
import datetime
import numpy as np
import os
import sys
# Environment Model
# sumoBinary = "/usr/local/bin/sumo"
sys.path.append(os.path.join('E:/sumo-0.32.0/', 'tools'))
sys.path.append('E:/sumo-0.32.0/')

sumoBinary = "sumo-gui"
traffic_path = "../data/map1.sumocfg"
sumoCmd = [sumoBinary, "-c", traffic_path, "--start"]  # The path to the sumo.cfg file
traci.start(sumoCmd)

def list_of_n_phases(TLIds):
    n_phases = []
    for light in TLIds:
        n_phases.append(int((len(traci.trafficlights.getRedYellowGreenState(light)) ** 0.5) * 2))
    return n_phases
def makemap(TLIds):
    maptlactions = []
    n_phases = list_of_n_phases(TLIds)
    for n_phase in n_phases:
        mapTemp = []
        if len(maptlactions) == 0:
            for i in range(n_phase):
                if i%2 == 0:
                    maptlactions.append([i])
        else:
            for state in maptlactions:
                for i in range(n_phase):
                    if i%2 == 0:
                        mapTemp.append(state+[i])
            maptlactions = mapTemp
    return maptlactions
def get_waittime():
    wait_time_map = {}
    for veh_id in traci.vehicle.getIDList():
        wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
    wait_temp = dict(wait_time_map)
    wait_sum_time = sum(wait_temp[x] for x in wait_temp)
    return wait_sum_time

tls = traci.trafficlight.getIDList()
# A = traci.edge.getIDCount()
D = traci.edge.getIDList()
EID = [v for v in D if 'edge'in v]
actionsMap = makemap(tls)
# traci.gui.screenshot(viewID=traci._gui.GuiDomain.DEFAULT_VIEW, filename="1.jpg")
for i in range(100):
    traci.simulationStep()
    action = np.random.randint(low=0, high=len(actionsMap))
    wait_time = get_waittime()
    lightsPhase = actionsMap[action]
    for light, index in zip(tls, range(len(tls))):
        traci.trafficlights.setPhase(light, lightsPhase[index])
    for iedge in EID:
        num = traci.edge.getLastStepVehicleNumber(iedge)
    S = traci.trafficlights.getRedYellowGreenState(tls[0])

    time.sleep(0.1)
    traci.gui.screenshot(viewID=traci._gui.GuiDomain.DEFAULT_VIEW, filename="1.jpg")

traci.close()