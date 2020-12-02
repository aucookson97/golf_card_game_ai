from enum import Enum
from card_deck import Card
from dqn_agent import DqnAgent
import os
import numpy as np

class PlayerType(Enum):
    Human = 0
    Computer = 1

class Player:

    def __init__(self, name, type_):
        self.score = 0
        self.name = name
        self.hand = None
        self.mask = None
        self.type = type_

        if self.type == PlayerType.Computer:
            self.agent = DqnAgent(train=False)

    def move(self, deck, top_card, verbose):
        if self.type == PlayerType.Human:
            self.display_hand()
            print('{}, Please Choose a # from the following options'.format(self.name))
            print ('\t 1: Draw a card from the deck')
            print ('\t 2: Pick up a {} from the pile'.format(top_card.name))
            selection = input('Selection: ') 
            if selection == '1':
                card = deck.draw()
                print ('You drew a {}'.format(card.name))
            else:
                card = top_card
            print ('What will you do with this {}?'.format(card.name))
            print ('Enter a Number (1-6) to replace that card, or 0 to put it back on the pile')
            selection = input('Selection: ')
            if selection == '0':
                return card
            else:
                position = int(selection) - 1
                old_card = self.hand[position]
                self.hand[position] = card
                self.mask[position] = 1
                return old_card
        else:

            # Pick up card from pile if it 0s a column or is <= 6
            draw = True
            for column in range(3):
                card1 = self.hand[column]
                card2 = self.hand[3 + column]
                if card1 != card2 and card1 != Card.Joker and card2 != Card.Joker and (top_card == card1 or top_card == card2):
                     draw = False
            if draw and top_card.value[1] <= 6:
                draw = False 
            
            if draw:
                card = deck.draw()
                if verbose:
                    print ('\t{} drew a {} from the deck.'.format(self.name, card.name))
            else:
                card = top_card
                if verbose:
                    print ('\t{} picked up a {} from the pile.'.format(self.name, card.name))  

            # Now decide what to do with that card
            state = self._get_dqn_state(top_card)
            action = self.agent.policy(state)

            if action == 0:
                if verbose:
                    print ('\t{} returned the {} to the pile.'.format(self.name, card.name))
                    self.display_hand()
                return card
            else:
                old_card = self.hand[action-1]
                self.hand[action-1] = card
                if verbose:
                    print ('\t{} replaced the {}{} at position {} with the {}.'.format(self.name, old_card.name, '' if self.mask[action-1] else '(hidden)', action, card.name))
                    self.display_hand()
                self.mask[action-1] = 1
                return old_card

    def _get_dqn_state(self, top_card):
        state = [top_card.value[1]]
        for i, card in enumerate(self.hand):
            if self.mask[i]:
                state.append(card.value[1])
            else:
                state.append(-1)
        return np.array(state)

    def flip_cards(self):
        if self.type == PlayerType.Human:
            selection = input('{}, Please Select Which Cards to flip, left-to-right beginning at 1 (i.e. \'1, 4\'): '.format(self.name)).replace(' ', '').split(',')
            self.mask[int(selection[0])-1] = 1
            self.mask[int(selection[1])-1] = 1
        else:
            self.mask[0] = 1
            self.mask[1] = 1

    def new_hand(self, cards):
        self.hand = cards
        self.mask = [0 for _ in range(len(self.hand))]

    def is_done(self):
        return all(self.mask)

    def get_hand_score(self):
        hand_score = 0
        if not self.hand:
            return None
        for column in range(3):
            card1 = self.hand[column]
            card2 = self.hand[3 + column]
            if card1 != card2 and card1 != Card.Joker and card2 != Card.Joker:
                hand_score += card1.value[0] + card2.value[0]
        return hand_score

    def display_hand(self):
        s = ''
        for i in range(len(self.hand)):
            if self.mask[i]:
                s += self.hand[i].name
                for _ in range(6 - len(self.hand[i].name)):
                    s += ' '
            else:
                s += '  ?   '
            if i == 2:
                print(s + '\n')
                s = ''
        print (s + '\n')
