from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class Hand:
    def __init__(self):
        self.cards = []
        self.softaces = 0
        self.hardaces = 0
        self.count = 0
        self.numSplit = 0

    def addCard(self, card):
        self.cards.append(card)
        if card == 'A':
            self.softaces += 1
            self.count += 11
        
        elif card == 'J' or card == 'Q' or card == 'K':
            self.count += 10
        
        else:
            self.count += int(card)

        if self.count > 21 and self.softaces > 0:
            self.hardaces += 1
            self.softaces -= 1
            self.count -= 10
    
    def removeCard(self):
        card = self.cards.pop()

        if card == 'A':
            if self.hardaces > 0:
                self.hardaces -= 1
                self.count -= 1
            else:
                self.softaces -= 1
                self.count -= 11

        elif card == 'J' or card == 'Q' or card == 'K':
            self.count -= 10

        else:
            self.count -= int(card)

def getCardValue(card: str) -> int:
    if card == 'A':
        return 11
    elif card == 'J' or card == 'Q' or card == 'K':
        return 10
    else:
        return int(card)

def Qstart():
    return [0, 0]

class BlackjackRLAgent:
    def __init__(
            self,
            learning_rate,
            learning_rate_decay,
            final_learning_rate,
            epsilon,
            epsilon_decay,
            final_epsilon,
            discount
            ):
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.final_learning_rate = final_learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount = discount

        self.q_table = defaultdict(Qstart) # 0 = hit, 1 = stand

    def getAction(self, obs: tuple[int, int, bool]):
        if np.random.random() < self.epsilon:
            choices = len(self.q_table[obs])
            return np.random.randint(0, choices)
        else:
            maxQValue = float('-inf')
            retAction = None
            for action, Qvalue in enumerate(self.q_table[obs]):
                if Qvalue > maxQValue:
                    maxQValue = Qvalue
                    retAction = action
            return retAction
        
    def update(self, obs, action, next_obs, reward, isDone):
        maxQValue = float('-inf')
        for Qvalue in self.q_table[next_obs]:
            if Qvalue > maxQValue:
                maxQValue = Qvalue
        futureQvalue = int(not isDone) * maxQValue
        sample = reward + self.discount * futureQvalue
        #print("sample: ", sample)
        #print("Before update: ", self.q_table[obs][action])
        self.q_table[obs][action] = (1 - self.learning_rate) * self.q_table[obs][action] + self.learning_rate * sample
        #print("After update: ", self.q_table[obs][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.final_epsilon)

    def decay_learning(self):
        self.learning_rate = max(self.learning_rate - self.learning_rate_decay, self.final_learning_rate)

def start_episode(agent: BlackjackRLAgent):
    deck = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    agent_hand = Hand()
    dealer_hand = Hand()

    for i in range(2):
        random_card = np.random.choice(deck)
        agent_hand.addCard(random_card)
    for i in range(2):
        random_card = np.random.choice(deck)
        dealer_hand.addCard(random_card)

    agent_count = agent_hand.count
    dealer_card = dealer_hand.cards[1]
    hasAce = (agent_hand.aces > 0) 

    if dealer_card == 'A':
        dealer_card = 11
    elif dealer_card == 'J' or dealer_card == 'Q' or dealer_card == 'K':
        dealer_card = 10
    else:
        dealer_card = int(dealer_card)

    cur_obs = (agent_count, dealer_card, hasAce)
    #print(cur_obs)
    done = False
    while not done:
        action = agent.getAction(cur_obs)
        match action:
            case 0: #hit
                random_card = np.random.choice(deck)
                agent_hand.addCard(random_card)
                agent_count = agent_hand.count
                hasAce = (agent_hand.softaces > 0) 
                next_obs = (agent_count, dealer_card, hasAce)
                #print("Hit: ", next_obs)
                if agent_hand.count > 21: 
                    done = True
                    reward = -2
                else:
                    reward = 0
                agent.update(cur_obs, action, next_obs, reward, done)
                cur_obs = next_obs
            case 1: #stand
                done = True
                #print("Stand: ", cur_obs)
                while (dealer_hand.count < 17 or (dealer_hand.count == 17 and dealer_hand.aces > 0)):
                    random_card = np.random.choice(deck)
                    dealer_hand.addCard(random_card)
                if (dealer_hand.count > 21 or dealer_hand.count < agent_hand.count):
                    reward = 2
                elif (dealer_hand.count > agent_hand.count):
                    reward = -2
                else:
                    reward = 0
                agent.update(cur_obs, action, cur_obs, reward, done)
    agent.decay_epsilon()
    agent.decay_learning()

def create_grid(agent: BlackjackRLAgent):
    policy = defaultdict(int)

    for obs, QvalueList in agent.q_table.items():
        action = None
        maxQvalue = float('-inf')
        for act, Qvalue in enumerate(QvalueList):
            if Qvalue > maxQvalue:
                maxQvalue = Qvalue
                action = act
        policy[obs] = action

    dealer_count, player_count = np.meshgrid(np.arange(2,12), np.arange(17,7,-1))

    policy_grid_noAce = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], False)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    dealer_count, player_count = np.meshgrid(np.arange(2,12), np.arange(20,12,-1))

    policy_grid_Ace = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], True)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    return (policy_grid_noAce, policy_grid_Ace)

def create_plot(grids):
    fig, axs = plt.subplots(2, 1)

    color_map = {
        0: np.array([255, 255, 255]),
        1: np.array([239, 197, 25])
    }

    policy_color_noAce = np.ndarray(shape=(grids[0].shape[0], grids[0].shape[1], 3), dtype=int)
    for i in range(grids[0].shape[0]):
        for j in range(grids[0].shape[1]):
            policy_color_noAce[i][j] = color_map[grids[0][i][j]]

    axs[0].imshow(policy_color_noAce)

    axs[0].set_xticks(list(range(10)), list(range(2, 11)) + ["A"])
    axs[0].set_yticks(list(range(10)), list(range(17, 7, -1)))
    axs[0].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs[0].set_xlabel("Dealer Upcard")
    axs[0].xaxis.set_label_position('top')
    axs[0].set_ylabel("Hard Totals")

    for i in range(10):
        for j in range(10):
            text = axs[0].text(j, i, grids[0][i, j], ha="center", va="center", color="black")

    policy_color_Ace = np.ndarray(shape=(grids[1].shape[0], grids[1].shape[1], 3), dtype=int)
    for i in range(grids[1].shape[0]):
        for j in range(grids[1].shape[1]):
            policy_color_Ace[i][j] = color_map[grids[1][i][j]]

    axs[1].imshow(policy_color_Ace)

    ytickslist = ["A," + str(i) for i in range(9, 1, -1)]

    axs[1].set_xticks(list(range(10)), list(range(2, 11)) + ["A"])
    axs[1].set_yticks(list(range(8)), ytickslist)
    axs[1].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs[1].set_ylabel("Soft Totals")

    for i in range(8):
        for j in range(10):
            text = axs[1].text(j, i, grids[1][i, j], ha="center", va="center", color="black")
    
    legend_elements = [
        Patch(facecolor="white", edgecolor="black", label="Hit"),
        Patch(facecolor=(239/255, 197/255, 25/255), edgecolor="black", label="Stand"),
    ]
    fig.legend(handles=legend_elements, loc='upper right')

    fig.tight_layout()
    plt.show()
        
if __name__ == "__main__":
    num_episodes = 1000000
    learning_rate_start = 0.1
    learning_rate_decay = learning_rate_start / num_episodes
    final_learning_rate = 0.001
    discount = 0.99
    epsilon_start = 1.0
    final_epsilon = 0.1
    epsilon_decay = epsilon_start / (num_episodes / 2)

    agent = BlackjackRLAgent(learning_rate_start, learning_rate_decay, final_learning_rate, epsilon_start, epsilon_decay, final_epsilon, discount)

    for episode in tqdm(range(num_episodes)):
        start_episode(agent)

    grids = create_grid(agent)
    create_plot(grids)
    print("Finished training!")