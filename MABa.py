import traci
import numpy as np
import pandas as pd
from collections import defaultdict

# === CONFIGURAÇÃO: Entradas e edges mapeados conforme SUMO === #
entrances = ['Avintes', 'Freixo', 'Oliveira_Douro', 'Antas', 'Boavista']
entrance_edge_map = {
    'Avintes': '54252',      # Substituir pelos IDs reais do net.xml!
    'Freixo': '1122620',
    'Oliveira_Douro': '30454',
    'Antas': '1250858',
    'Boavista': '6075'
}

sumocfg_file = "planner.sumo.cfg.xml"  # ou net.net.xml ? 

class UCB_MAB:
    def __init__(self, arms):
        self.arms = arms
        self.counts = {arm: 1e-3 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}

    def select_arm(self):
        total_counts = sum(self.counts.values())
        ucb_values = {
            arm: self.values[arm] + np.sqrt(2 * np.log(total_counts)/self.counts[arm])
            for arm in self.arms
        }
        return max(ucb_values, key=ucb_values.get)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n-1)/n)*value + (1/n)*reward

def inject_vehicles(edge_id, n_vehicles, step):
    for i in range(n_vehicles):
        veh_id = f"banditVeh_{edge_id}_{step}_{i}"
        route_id = f"{edge_id}-main"  # Precisas mapear/definir as rotas no routes.xml!
        traci.vehicle.add(veh_id, routeID=route_id, depart=step + i)
        traci.vehicle.moveTo(veh_id, edge_id)

def get_reward(volume_real, volume_simulado):
    # Pode usar RMSE, MAE, etc. Aqui está RMSE:
    volume_real = np.array(volume_real)
    volume_simulado = np.array(volume_simulado)
    return -np.sqrt(np.mean((volume_real - volume_simulado) ** 2))

def run_bandit_episode(mab, volume_real, entrance_edge_map, max_vehicles=100):
    traci.start(["sumo-gui", "-c", sumocfg_file])
    chosen_arm = mab.select_arm()
    n_vehicles = int(max_vehicles)
    inject_vehicles(entrance_edge_map[chosen_arm], n_vehicles, step=0)
    sim_volumes = defaultdict(int)
    # Exemplo de ciclo de simulação de 5 minutos
    for t in range(300):
        traci.simulationStep()
        for arm in mab.arms:
            edge = entrance_edge_map[arm]
            sim_volumes[arm] += traci.edge.getLastStepVehicleNumber(edge)
    traci.close()
    reward = get_reward([volume_real[arm] for arm in mab.arms], [sim_volumes[arm] for arm in mab.arms])
    mab.update(chosen_arm, reward)
# --- Exemplo real ---
# Carrega volumes reais conforme CSV de sensores + mapeamento de entrada
sensor_data = pd.read_csv("sample.csv", encoding="iso-8859-1")
volume_real = {arm: sensor_data[sensor_data['EQUIPMENTID'] == int(x)]['TOTAL_VOLUME'].mean() for arm, x in entrance_edge_map.items()}
mab = UCB_MAB(arms=entrances)
run_bandit_episode(mab, volume_real, entrance_edge_map, max_vehicles=100)
