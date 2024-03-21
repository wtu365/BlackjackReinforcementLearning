from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class Hand:
    def __init__(self):
        self.cards = []
        self.aces = 0
        self.count = 0

    def addCard(self, card):
        self.cards.append(card)
        if card == 'A':
            self.aces += 1
            self.count += 11
        
        elif card == 'J' or card == 'Q' or card == 'K':
            self.count += 10
        
        else:
            self.count += int(card)

        if self.count > 21 and self.aces > 0:
            self.aces -= 1
            self.count -= 10

def Qstart():
    return [0, 0]

class BlackjackRLAgent:
    def __init__(
            self,
            learning_rate,
            epsilon,
            epsilon_decay,
            final_epsilon,
            discount
            ):
        self.learning_rate = learning_rate
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
                hasAce = (agent_hand.aces > 0) 
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

def create_grid(agent: BlackjackRLAgent, usable_ace):
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

    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    return policy_grid

def create_plot(policy_grid):
    fig, ax = plt.subplots()

    color_map = {
        0: np.array([255, 255, 255]),
        1: np.array([239, 197, 25])
    }
    policy_color = np.ndarray(shape=(policy_grid.shape[0], policy_grid.shape[1], 3), dtype=int)
    for i in range(policy_grid.shape[0]):
        for j in range(policy_grid.shape[1]):
            policy_color[i][j] = color_map[policy_grid[i][j]]

    ax.imshow(policy_color)

    ax.set_xticks(list(range(10)), list(range(2, 11)) + ["A"])
    ax.set_yticks(list(range(10)), list(range(17, 7, -1)))
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_xlabel("Dealer Upcard")
    ax.xaxis.set_label_position('top')
    ax.set_ylabel("Player sum")

    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, policy_grid[i, j], ha="center", va="center", color="black")
    
    legend_elements = [
        Patch(facecolor="white", edgecolor="black", label="Hit"),
        Patch(facecolor=(239/255, 197/255, 25/255), edgecolor="black", label="Stick"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

    fig.tight_layout()
    plt.show()

def play_blackjack():
    deck = {'A':4, '2':4, '3':4, '4':4, '5':4, '6':4, '7':4, '8':4, '9':4, '10':4, 'J':4, 'Q':4, 'K':4}
    player = Hand()
    dealer = Hand()

    for i in range(2):
        random_card = random.choice(list(deck.keys()))
        player.addCard(random_card)
        deck[random_card] -= 1
        if deck[random_card] == 0:
            deck.pop(random_card)
    for i in range(2):
        random_card = random.choice(list(deck.keys()))
        dealer.addCard(random_card)
        deck[random_card] -= 1
        if deck[random_card] == 0:
            deck.pop(random_card)
    
    player_action = ""
    actions = ["hit", "stand"]

    print("Player: ")
    print(player.cards)
    print("Dealer Card: ")
    print(dealer.cards[1])

    if len(player.cards) == 2 and player.count == 21:
        print("Congrtuations! You got a blackjack!")
        return

    while True:
        player_action = input("Choose from one of the following actions [%s]: " % " ".join(actions)).lower()
        match player_action:
            case "hit":
                random_card = random.choice(list(deck.keys()))
                player.addCard(random_card)
                deck[random_card] -= 1
                if deck[random_card] == 0:
                    deck.pop(random_card)
            case "stand":
                break
            case __:
                print("Not a valid input. Please try again.")
                continue
        print("Player: ")
        print(player.cards)
        print("Dealer Card: ")
        print(dealer.cards[1])

        if (player.count > 21):
            print("Sorry! Your hand is over 21 and has busted.")
            return
    while (dealer.count < 17 or (dealer.count == 17 and dealer.aces > 0)):
        random_card = random.choice(list(deck.keys()))
        dealer.addCard(random_card)
        deck[random_card] -= 1
        if deck[random_card] == 0:
            deck.pop(random_card)
    
    print("Player: ")
    print(player.cards)
    print("Dealer: ")
    print(dealer.cards)
    if (dealer.count > 21):
        print("Congratulations! The dealer has busted and you have won!")
        return
    elif (dealer.count > player.count):
        print("Unfortunately, your hand is lower than the dealer's hand. You have lost.")
        return
    elif (dealer.count < player.count):
        print("Congratulations! Your hand is higher than the dealer's hand. You have won!")
        return
    else:
        print("It's a tie! You will get your bet back.")
        return
        
if __name__ == "__main__":
    learning_rate = 0.01
    num_episodes = 100000
    discount = 0.99
    epsilon_start = 1.0
    final_epsilon = 0.1
    epsilon_decay = epsilon_start / (num_episodes / 2)

    agent = BlackjackRLAgent(learning_rate, epsilon_start, epsilon_decay, final_epsilon, discount)

    for episode in tqdm(range(num_episodes)):
        start_episode(agent)

    policy_grid = create_grid(agent, False)
    create_plot(policy_grid)
    print("Finished training!")