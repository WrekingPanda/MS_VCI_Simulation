import numpy as np

class BanditAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Base Multi-Armed Bandit Agent (Epsilon-Greedy / Q-Learning style).
        
        Args:
            action_space (list): List of possible actions.
            learning_rate (float): Alpha.
            discount_factor (float): Gamma (if contextual/state-based).
            epsilon (float): Exploration rate.
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-table: Map state -> {action: value}
        # For simple MAB, state might be constant "global".
        self.q_table = {} 

    def get_q(self, state, action):
        return self.q_table.get(state, {}).get(action, 0.0)

    def update(self, state, action, reward, next_state=None):
        """
        Update Q-value: Q(s,a) = Q(s,a) + alpha * (r - Q(s,a))
        (Simple Bandit version, can be extended to Q-learning if next_state max Q is used)
        """
        current_q = self.get_q(state, action)
        
        # Standard Bandit update (running average of rewards if alpha=1/n, or weighted if constant alpha)
        # MAB equation: Q_{k+1} = Q_k + alpha * (R_k - Q_k)
        new_q = current_q + self.learning_rate * (reward - current_q)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q

    def choose_action(self, state):
        """
        Epsilon-greedy selection.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
            
        # Greedy
        if state not in self.q_table or not self.q_table[state]:
            return np.random.choice(self.action_space)
            
        actions_values = self.q_table[state]
        # Find max value
        max_val = max(actions_values.values())
        # All actions with that max value (break ties randomly)
        best_actions = [a for a, v in actions_values.items() if v == max_val]
        
        # If we haven't explored all actions for this state, should we prioritize them?
        # Standard implementation: just choose best from known, or random if empty.
        # But wait, if some actions are never tried, they are not in dict.
        # Let's ensure we default to 0 for untried actions if they are not in dict.
        
        curr_best_actions = []
        curr_max = -float('inf')
        
        for a in self.action_space:
            val = actions_values.get(a, 0.0) # Default expectation 0
            if val > curr_max:
                curr_max = val
                curr_best_actions = [a]
            elif val == curr_max:
                curr_best_actions.append(a)
                
        return np.random.choice(curr_best_actions)


class EntranceAgent(BanditAgent):
    """
    Approach A: One Agent per Entrance.
    Controls the number of vehicles (flow rate) or route distribution entering.
    """
    def __init__(self, entrance_id, action_space, **kwargs):
        super().__init__(action_space, **kwargs)
        self.entrance_id = entrance_id

class VehicleAgent(BanditAgent):
    """
    Approach B: One Agent per Vehicle.
    Decides the route for a specific vehicle.
    Arguments:
        action_space: List of Route IDs.
    """
    def __init__(self, vehicle_id, action_space, **kwargs):
        super().__init__(action_space, **kwargs)
        self.vehicle_id = vehicle_id
