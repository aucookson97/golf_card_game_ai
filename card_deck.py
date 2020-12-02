from enum import Enum
import random

class Card(Enum):  
    # Value, Index 
    Joker = (-1, 0)
    Ace = (1, 1)
    Two = (-2, 2)
    Three = (3, 3)
    Four = (4, 4)
    Five = (5, 5)
    Six = (6, 6)
    Seven = (7, 7)
    Eight = (8, 8)
    Nine = (9, 9)
    Ten = (10, 10)
    Jack = (10, 11)
    Queen = (10, 12)
    King = (0, 13)

class Deck:
    def __init__(self):
        self.deck = None
        self.shuffle()

    def shuffle(self):
        """ Restarts and Shuffles deck
        """
        self.deck = []
        for card in Card:
            if card != Card.Joker:
                for _ in range(4):
                    self.deck.append(card)
            else:
                for _ in range(2):
                    self.deck.append(card)
        random.shuffle(self.deck)

    def draw_n(self, n):
        """ Draw n cards from the top of the deck
        """
        if len(self.deck) > n:
            cards = self.deck[len(self.deck)-n:]
            self.deck = self.deck[:len(self.deck)-n]
            return cards

    def draw(self):
        """ Draw one card from the top of the deck
        """
        if len(self.deck) > 0:
            return self.deck.pop()
        else:
            return -1