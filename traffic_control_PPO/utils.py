import traci

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
