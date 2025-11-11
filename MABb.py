import traci
import numpy as np

# Definir OD pairs (do TAZs e edges para o teu net.xml/trips.xml)
taz_pairs = [('62','47'), ('108','126'), ('28','115'), ('129','166'), ('158','20'), ('40','71'), ('73','33'), ('36','25')]
routes_by_taz = {
    ('62','47'): "route_id_62_47",   # Mapear conforme routes.xml
    ('108','126'): "route_id_108_126",
    # ...
}

sumocfg_file = "planner.sumo.cfg.xml"  # ou net.net.xml ? 

class ThompsonBandit:
    def __init__(self, arms):
        self.arms = arms
        self.successes = {arm: 1 for arm in arms}
        self.failures = {arm: 1 for arm in arms}

    def select_arm(self):
        samples = {arm: np.random.beta(self.successes[arm], self.failures[arm]) for arm in self.arms}
        return max(samples, key=samples.get)

    def update(self, chosen_arm, reward):
        if reward > 0:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1

def inject_vehicle_route(taz_in, taz_out, step):
    v_id = f"ODVeh_{taz_in}_{taz_out}_{step}"
    route_id = routes_by_taz[(taz_in, taz_out)]
    traci.vehicle.add(v_id, routeID=route_id, depart=step)
    # opcional: mover para edge inicial traci.vehicle.moveTo(v_id, start_edge_id)

def get_reward(sim_volume, real_volume):
    return 1.0 - abs(sim_volume-real_volume)/max(real_volume,1.0)

def run_od_bandit_episode(bandit, sensor_real_data, n_vehicles=20):
    traci.start(["sumo", "-c", sumocfg_file])
    chosen_od = bandit.select_arm()
    taz_in, taz_out = chosen_od
    for step in range(n_vehicles):
        inject_vehicle_route(taz_in, taz_out, step)
    sim_volume = traci.edge.getLastStepVehicleNumber(routes_by_taz[chosen_od])  # ou no edge de entrada/saída específico
    real_volume = sensor_real_data.get((taz_in,taz_out), 10)
    reward = get_reward(sim_volume, real_volume)
    bandit.update(chosen_od, reward)
    traci.close()
# --- Exemplo real ---
sensor_real_data = {od: 15 for od in taz_pairs} # usar dados reais dos sensores
bandit = ThompsonBandit(arms=taz_pairs)
run_od_bandit_episode(bandit, sensor_real_data, n_vehicles=20)
