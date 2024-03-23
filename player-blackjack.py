import random

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
    play_game = input("Would you like to play a round of simple blackjack? Write y for yes and anything else for no: ").lower()
    if (play_game == 'y'):
        play_blackjack()
        play_game = input("Would you like to play again? Write y for yes and anything else for no: ").lower()
    print("Thanks for playing!")
