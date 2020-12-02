from card_deck import Card, Deck
from random import sample
import numpy as np

class GolfGame:

    def __init__(self, num_rounds_per_episode=10):
        self.deck = Deck()
        self.reset()
        self.rounds_per_episode = num_rounds_per_episode

    def reset(self):
        self.round = -1
        self.new_hand()
        return self._get_state()

    def new_hand(self):
        self.deck.shuffle()
        self.hand = self.deck.draw_n(6)
        self.top_card = self.deck.draw()
        self.mask = [0, 0, 0, 0, 0, 0]
        flip = sample(range(0, 5), 2)
        self.mask[flip[0]] = 1
        self.mask[flip[1]] = 1

        self.round += 1

    def step(self, action):
        initial_score = self._get_hand_score()

        # Make sure we dont run out of cards
        if len(self.deck.deck) == 0:
            self.deck.shuffle()

        if action == 0:     # Put card back
            self.top_card = self.deck.draw()
        else:     # Replace card
            self.hand[action-1] = self.top_card
            self.mask[action-1] = 1
        next_state = self._get_state()
        reward = initial_score - self._get_hand_score() 
        if all(self.mask):      # Every card is flipped over, restart
            self.new_hand()

        done = (self.round >= self.rounds_per_episode)

        return next_state, reward, done

    def _get_state(self):
        state = [self.top_card.value[1]]
        for i, card in enumerate(self.hand):
            if self.mask[i]:
                state.append(card.value[1])
            else:
                state.append(-1)
        return np.array(state)

    def _get_hand_score(self):
        hand_score = 0
        if not self.hand:
            return None
        for column in range(3):
            card1 = self.hand[column]
            card2 = self.hand[3 + column]
            if card1 != card2 and card1 != Card.Joker and card2 != Card.Joker:
                hand_score += card1.value[0] + card2.value[0]
        return hand_score