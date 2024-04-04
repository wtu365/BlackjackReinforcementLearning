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

def Qstart(obs: tuple[int, int, bool]):
    array = [0, 0, 0, 0, 0]
    return array

class keydefaultdict(defaultdict): # Subclass for defaultdict in order to be able to create different sized lists for different states.
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

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

        self.q_table = keydefaultdict(Qstart) # 0 = hit, 1 = stand, 2 = double, 3 = surrender, 4 = split

    def getAction(self, obs: tuple[int, int, bool], canDouble_Surrender: bool, canSplit: bool):
        if np.random.random() < self.epsilon:
            choices = 2 + 2 * int(canDouble_Surrender) + int(canSplit)
            return np.random.randint(0, choices)
        else:
            choices = 2 + 2 * int(canDouble_Surrender) + int(canSplit)
            maxQValue = float('-inf')
            retAction = None
            for action, Qvalue in enumerate(self.q_table[obs]):
                if Qvalue > maxQValue and action < choices:
                    maxQValue = Qvalue
                    retAction = action
            return retAction
        
    def update(self, obs, action, next_obs, reward, isDone, canDouble=False, canSplit=False):
        maxQValue = float('-inf')
        maxAction = 2 + 2 * int(canDouble) + int(canSplit)
        for act, Qvalue in enumerate(self.q_table[next_obs]):
            if Qvalue > maxQValue and act < maxAction:
                maxQValue = Qvalue
        futureQvalue = int(not isDone) * maxQValue
        sample = reward + self.discount * futureQvalue
        self.q_table[obs][action] = (1 - self.learning_rate) * self.q_table[obs][action] + self.learning_rate * sample

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.final_epsilon)

    def decay_learning(self):
        self.learning_rate = max(self.learning_rate - self.learning_rate_decay, self.final_learning_rate)

def start_episode(agent: BlackjackRLAgent, agent_start_hand: Hand, dealer_start_hand: Hand, isSplitHand: bool):
    deck = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    agent_hand = agent_start_hand
    dealer_hand = dealer_start_hand

    while len(agent_hand.cards) < 2:
        random_card = np.random.choice(deck)
        agent_hand.addCard(random_card)
    while len(dealer_hand.cards) < 2:
        random_card = np.random.choice(deck)
        dealer_hand.addCard(random_card)

    agent_count = agent_hand.count
    dealer_card = dealer_hand.cards[1]
    hasAce = (agent_hand.softaces > 0) 
    canDouble_Surrender = True
    if (getCardValue(agent_hand.cards[0]) == getCardValue(agent_hand.cards[1])):
        canSplit = True
    else:
        canSplit = False

    if dealer_card == 'A':
        dealer_card = 11
    elif dealer_card == 'J' or dealer_card == 'Q' or dealer_card == 'K':
        dealer_card = 10
    else:
        dealer_card = int(dealer_card)

    cur_obs = (agent_count, dealer_card, hasAce)
    done = False

    if len(dealer_hand.cards) == 2 and dealer_hand.count == 21:
        done = True
        if len(agent_hand.cards) == 2 and agent_hand.count == 21:
            reward = 0
        else:
            reward = -2
    elif len(agent_hand.cards) == 2 and agent_hand.count == 21:
        done = True
        reward = 3

    while not done:
        action = agent.getAction(cur_obs, canDouble_Surrender, canSplit) # 0 = hit, 1 = stand, 2 = double, 3 = surrender, 4 = split
        match action:
            case 0: #hit
                random_card = np.random.choice(deck)
                agent_hand.addCard(random_card)
                agent_count = agent_hand.count
                hasAce = (agent_hand.softaces > 0)
                canDouble_Surrender = False
                canSplit = False
                next_obs = (agent_count, dealer_card, hasAce)
                if agent_hand.count > 21: 
                    done = True
                    reward = -2
                else:
                    reward = 0
                agent.update(cur_obs, action, next_obs, reward, done, canDouble_Surrender, canSplit)
                cur_obs = next_obs
            case 1: #stand
                done = True
                while (dealer_hand.count < 17 or (dealer_hand.count == 17 and dealer_hand.softaces > 0)):
                    random_card = np.random.choice(deck)
                    dealer_hand.addCard(random_card)
                if (dealer_hand.count > 21 or dealer_hand.count < agent_hand.count):
                    reward = 2
                elif (dealer_hand.count > agent_hand.count):
                    reward = -2
                else:
                    reward = 0
                agent.update(cur_obs, action, cur_obs, reward, done)
            case 2: #double
                done = True
                random_card = np.random.choice(deck)
                agent_hand.addCard(random_card)
                if agent_hand.count > 21: 
                    reward = -4
                else:
                    while (dealer_hand.count < 17 or (dealer_hand.count == 17 and dealer_hand.softaces > 0)):
                        random_card = np.random.choice(deck)
                        dealer_hand.addCard(random_card)
                    if (dealer_hand.count > 21 or dealer_hand.count < agent_hand.count):
                        reward = 4
                    elif (dealer_hand.count > agent_hand.count):
                        reward = -4
                    else:
                        reward = 0
                agent.update(cur_obs, action, cur_obs, reward, done)
            case 3: #surrender
                if isSplitHand:
                    continue
                done = True
                reward = -1
                agent.update(cur_obs, action, cur_obs, reward, done)
            case 4: #split
                done = True
                split_hand = Hand()
                split_hand.addCard(agent_hand.cards[1])
                agent_hand.removeCard()
                reward = start_episode(agent, agent_hand, dealer_hand, True) + start_episode(agent, split_hand, dealer_hand, True)
                agent.update(cur_obs, action, cur_obs, reward, done)
    if isSplitHand:
        return reward
    agent.decay_epsilon()
    agent.decay_learning()

def create_grid(agent: BlackjackRLAgent):
    policy_nosur_nosplit = defaultdict(int)
    policy_withsurrender = defaultdict(int)
    policy_allactions = defaultdict(int)

    for obs, QvalueList in agent.q_table.items():
        action = None
        maxQvalue = float('-inf')
        for act, Qvalue in enumerate(QvalueList):
            if Qvalue > maxQvalue and act < 3:
                maxQvalue = Qvalue
                action = act
        policy_nosur_nosplit[obs] = action

    for obs, QvalueList in agent.q_table.items():
        action = None
        maxQvalue = float('-inf')
        for act, Qvalue in enumerate(QvalueList):
            if Qvalue > maxQvalue and act < 4:
                maxQvalue = Qvalue
                action = act
        policy_withsurrender[obs] = action

    for obs, QvalueList in agent.q_table.items():
        action = None
        maxQvalue = float('-inf')
        for act, Qvalue in enumerate(QvalueList):
            if Qvalue > maxQvalue:
                maxQvalue = Qvalue
                action = act
        policy_allactions[obs] = action

    dealer_count, player_count = np.meshgrid(np.arange(2,12), np.arange(20,2,-2))

    policy_grid_split_1 = np.apply_along_axis(
        lambda obs: 1 if policy_allactions[(obs[0], obs[1], False)] == 4 else 0,
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    dealer_count, player_count = np.meshgrid(np.arange(2,12), np.arange(12,13))

    policy_grid_split_2 = np.apply_along_axis(
        lambda obs: 1 if policy_allactions[(obs[0], obs[1], True)] == 4 else 0,
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    policy_grid_split = np.concatenate((policy_grid_split_2, policy_grid_split_1), axis=0)

    dealer_count, player_count = np.meshgrid(np.arange(2,12), np.arange(17,13,-1))

    policy_grid_surrender = np.apply_along_axis(
        lambda obs: 1 if policy_withsurrender[(obs[0], obs[1], False)] == 3 else 0,
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    dealer_count, player_count = np.meshgrid(np.arange(2,12), np.arange(17,7,-1))

    policy_grid_noAce = np.apply_along_axis(
        lambda obs: policy_nosur_nosplit[(obs[0], obs[1], False)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    dealer_count, player_count = np.meshgrid(np.arange(2,12), np.arange(20,12,-1))

    policy_grid_Ace = np.apply_along_axis(
        lambda obs: policy_nosur_nosplit[(obs[0], obs[1], True)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    return (policy_grid_noAce, policy_grid_Ace, policy_grid_split, policy_grid_surrender)

def create_plot(grids):
    fig, axs = plt.subplots(2, 2)

    color_map = {
        0: np.array([255, 255, 255]),
        1: np.array([239, 197, 25]),
        2: np.array([36, 182, 112])
    }

    color_map_split_surrender = {
        0: np.array([255, 255, 255]),
        1: np.array([36, 182, 112])
    }

    policy_color_noAce = np.ndarray(shape=(grids[0].shape[0], grids[0].shape[1], 3), dtype=int)
    for i in range(grids[0].shape[0]):
        for j in range(grids[0].shape[1]):
            policy_color_noAce[i][j] = color_map[grids[0][i][j]]

    axs[0][0].imshow(policy_color_noAce)

    axs[0][0].set_xticks(list(range(10)), list(range(2, 11)) + ["A"])
    axs[0][0].set_yticks(list(range(10)), list(range(17, 7, -1)))
    axs[0][0].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs[0][0].set_xlabel("Dealer Upcard")
    axs[0][0].xaxis.set_label_position('top')
    axs[0][0].set_ylabel("Hard Totals")

    for i in range(10):
        for j in range(10):
            text = axs[0][0].text(j, i, grids[0][i, j], ha="center", va="center", color="black")

    policy_color_Ace = np.ndarray(shape=(grids[1].shape[0], grids[1].shape[1], 3), dtype=int)
    for i in range(grids[1].shape[0]):
        for j in range(grids[1].shape[1]):
            policy_color_Ace[i][j] = color_map[grids[1][i][j]]

    axs[1][0].imshow(policy_color_Ace)

    ytickslist = ["A," + str(i) for i in range(9, 1, -1)]

    axs[1][0].set_xticks(list(range(10)), list(range(2, 11)) + ["A"])
    axs[1][0].set_yticks(list(range(8)), ytickslist)
    axs[1][0].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs[1][0].set_ylabel("Soft Totals")

    for i in range(8):
        for j in range(10):
            text = axs[1][0].text(j, i, grids[1][i, j], ha="center", va="center", color="black")

    policy_color_Split = np.ndarray(shape=(grids[2].shape[0], grids[2].shape[1], 3), dtype=int)
    for i in range(grids[2].shape[0]):
        for j in range(grids[2].shape[1]):
            policy_color_Split[i][j] = color_map_split_surrender[grids[2][i][j]]

    axs[0][1].imshow(policy_color_Split)

    ytickslist = ["A, A"] + [str(i) + ", " + str(i) for i in range(10, 1, -1)]

    axs[0][1].set_xticks(list(range(10)), list(range(2, 11)) + ["A"])
    axs[0][1].set_yticks(list(range(10)), ytickslist)
    axs[0][1].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs[0][1].set_xlabel("Dealer Upcard")
    axs[0][1].xaxis.set_label_position('top')
    axs[0][1].set_ylabel("Pair Splitting")

    for i in range(10):
        for j in range(10):
            text = axs[0][1].text(j, i, grids[2][i, j], ha="center", va="center", color="black")
    
    policy_color_Surrender = np.ndarray(shape=(grids[3].shape[0], grids[3].shape[1], 3), dtype=int)
    for i in range(grids[3].shape[0]):
        for j in range(grids[3].shape[1]):
            policy_color_Surrender[i][j] = color_map_split_surrender[grids[3][i][j]]

    axs[1][1].imshow(policy_color_Surrender)

    axs[1][1].set_xticks(list(range(10)), list(range(2, 11)) + ["A"])
    axs[1][1].set_yticks(list(range(4)), list(range(17, 13, -1)))
    axs[1][1].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axs[1][1].set_ylabel("Surrender")

    for i in range(4):
        for j in range(10):
            text = axs[1][1].text(j, i, grids[3][i, j], ha="center", va="center", color="black")

    legend_elements = [
        Patch(facecolor="white", edgecolor="black", label="Hit"),
        Patch(facecolor=(239/255, 197/255, 25/255), edgecolor="black", label="Stand"),
        Patch(facecolor=(36/255, 182/255, 112/255), edgecolor="black", label="Double")
    ]

    legend_elements_split = [
        Patch(facecolor="white", edgecolor="black", label="No Split"),
        Patch(facecolor=(36/255, 182/255, 112/255), edgecolor="black", label="Split")
    ]

    legend_elements_surrender = [
        Patch(facecolor="white", edgecolor="black", label="No Sur"),
        Patch(facecolor=(36/255, 182/255, 112/255), edgecolor="black", label="Sur")
    ]

    axs[0][0].legend(handles=legend_elements, bbox_to_anchor=(1.5, 1))
    axs[0][1].legend(handles=legend_elements_split, bbox_to_anchor=(1.5, 1))
    axs[1][1].legend(handles=legend_elements_surrender, bbox_to_anchor=(1.3, 1))

    fig.tight_layout()
    plt.show()
        
if __name__ == "__main__":
    num_episodes = 10000000
    learning_rate_start = 0.0005
    learning_rate_decay = learning_rate_start / num_episodes 
    final_learning_rate = 0.0001
    discount = 0.95
    epsilon_start = 1.0
    final_epsilon = 0.1
    epsilon_decay = epsilon_start / (num_episodes / 2)

    agent = BlackjackRLAgent(learning_rate_start, learning_rate_decay, final_learning_rate, epsilon_start, epsilon_decay, final_epsilon, discount)

    for episode in tqdm(range(num_episodes)):
        start_episode(agent, Hand(), Hand(), False)

    grids = create_grid(agent)
    create_plot(grids)

    print("Finished training!")