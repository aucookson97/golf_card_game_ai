import numpy as np
from card_deck import Deck, Card
from player import Player, PlayerType

class Game:
    def __init__(self, players=None, num_rounds=9, verbose=True):
        if not players:
            players = [('AI1', PlayerType.Computer), ('AI1', PlayerType.Computer)]
        self.verbose = verbose
        self.players = []
        for player in players:
            name, type_ = player
            self.players.append(Player(name, type_))
        self.round = 1
        self.num_rounds = num_rounds
        self.deck = Deck()
        self.top = None
        self.first_player = 0

    def run(self):
        while self.round <= self.num_rounds:
            if self.verbose:
                print ('Round {}\n'.format(self.round))
            self._new_round()

            # Flip over initial cards
            for player in self.players:
                player.flip_cards()
                player.display_hand()

            # Go around the table until someone flips all of their cards over
            current_player = self.first_player
            playing = True
            while playing:
                if self.verbose:
                    print ('{}\'s Turn'.format(self.players[current_player].name))
                self.top = self.players[current_player].move(self.deck, self.top, self.verbose)

                if self.players[current_player].is_done():
                    playing = False
                current_player = (current_player + 1) % len(self.players)
                

            # When round is over, add hand score to player scores and increment starting player
            for player in self.players:    
                player.score += player.get_hand_score()
            self.first_player = (self.first_player + 1) % len(self.players)
            self.round += 1

        lowest_score = 10000
        winner = ''
        if self.verbose:
            print ('Final Hands: ')
        for player in self.players:
            player.mask = [1, 1, 1, 1, 1, 1]
            if self.verbose:
                print ('Player: {}'.format(player.name))
            player.display_hand()
        if self.verbose:
            print ('Final Scores:')
        avg_score = 0
        for player in self.players:
            if self.verbose:
                print ('\t {}: {}'.format(player.name, player.score))
            avg_score += player.score
            if player.score < lowest_score:
                winner = player.name
                lowest_score = player.score
            elif player.score == lowest_score:
                winner += ' and ' + player.name
        if self.verbose:
            print ('The winner is... {}!'.format(winner))

        avg_score /= (len(self.players) * self.num_rounds)
        return avg_score
    def _new_round(self):
        self.deck.shuffle()
        for player in self.players:
            player.new_hand(self.deck.draw_n(6))
        self.top = self.deck.draw()

if __name__=="__main__":
    players = [('Aidan', PlayerType.Computer), ('Bob', PlayerType.Computer)]
    game = Game(players, num_rounds=10)
    avg_score = game.run()