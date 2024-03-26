import random

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

def play_blackjack(player_hand: Hand, dealer_hand: Hand):
    deck = {'A':24, '2':24, '3':24, '4':24, '5':24, '6':24, '7':24, '8':24, '9':24, '10':24, 'J':24, 'Q':24, 'K':24}
    player = player_hand
    dealer = dealer_hand

    while len(player.cards) < 2:
        random_card = random.choice(list(deck.keys()))
        player.addCard(random_card)
        deck[random_card] -= 1
        if deck[random_card] == 0:
            deck.pop(random_card)
    while len(dealer.cards) < 2:
        random_card = random.choice(list(deck.keys()))
        dealer.addCard(random_card)
        deck[random_card] -= 1
        if deck[random_card] == 0:
            deck.pop(random_card)
    
    player_action = ""
    actions = ["hit", "stand", "double", "surrender"]

    if (getCardValue(player.cards[0]) == getCardValue(player.cards[1])):
        actions.append("split")

    print("Player: ")
    print(player.cards)

    if len(dealer.cards) == 2 and dealer.count == 21:
        print("Dealer Blackjack: ")
        print(dealer.cards)
        if len(player.cards) == 2 and player.count == 21:
            print("Both you and the dealer have a blackjack, so it is a tie!")
            return
        else:
            print("Unfortunately, the dealer has a blackjack and you do not. You lose!") 
            return

    print("Dealer Card: ")
    print(dealer.cards[1])

    if len(player.cards) == 2 and player.count == 21:
        print("Congrtuations! You got a blackjack!")
        return

    while True:
        player_action = input("Choose from one of the following actions [%s]: " % " ".join(actions)).lower()
        match player_action:
            case "hit":
                if len(player.cards) == 2:
                    if "split" in actions:
                        actions.remove("split")
                    actions.remove("double")
                    actions.remove("surrender")
                random_card = random.choice(list(deck.keys()))
                player.addCard(random_card)
                deck[random_card] -= 1
                if deck[random_card] == 0:
                    deck.pop(random_card)
            case "stand":
                break
            case "double":
                if "double" not in actions:
                    print("Not a valid input. Please try again.")
                    continue
                print("You have doubled your bet.")
                random_card = random.choice(list(deck.keys()))
                player.addCard(random_card)
                deck[random_card] -= 1
                if deck[random_card] == 0:
                    deck.pop(random_card)
                break
            case "split":
                if "split" not in actions:
                    print("Not a valid input. Please try again.")
                    continue
                split_hand = Hand()
                split_hand.addCard(player.cards[1])
                player.removeCard()
                print("Split Hand 1:")
                play_blackjack(player, dealer)
                print("Split Hand 2:")
                play_blackjack(split_hand, dealer)
                return
            case "surrender":
                if "surrender" not in actions:
                    print("Not a valid input. Please try again.")
                    continue
                print("You have decided to surrender. You will get half your bet back.")
                return
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
    while (dealer.count < 17 or (dealer.count == 17 and dealer.softaces > 0)):
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
    while (play_game == 'y'):
        play_blackjack(Hand(), Hand())
        play_game = input("Would you like to play again? Write y for yes and anything else for no: ").lower()
    print("Thanks for playing!")
