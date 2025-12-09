import random

# -----------------------
# Card values
# -----------------------
VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 10, 'Q': 10, 'K': 10,
    'A': 11
}


class Deck:
    def __init__(self, num_decks: int = 1):
        ranks = list(VALUES.keys())
        suits = ["H", "D", "C", "S"]

        self.cards = []
        for _ in range(num_decks):
            self.cards.extend((r, s) for r in ranks for s in suits)

        random.shuffle(self.cards)

    """
    def found_card(self, card: str):
        try:
            self.cards.remove(card)
        except:
            ...

    def draw(self):
        if not self.cards:
            raise RuntimeError("Tried to draw from empty deck")
        return self.cards.pop()
    """
    def size(self) -> int:
        return len(self.cards)

    def bust_probability(self, current_value: int) -> float:
        """Probability that drawing ONE more card will bust this value."""
        if not self.cards:
            return 1.0

        busting = 0
        for rank, suit in self.cards:
            if current_value + VALUES[rank] > 21:
                busting += 1

        return busting / len(self.cards)


# -----------------------
# Hand Logic
# -----------------------
"""
def hand_value(hand) -> int:
    total = sum(VALUES[r] for (r, s) in hand)
    aces = sum(r == 'A' for (r, s) in hand)

    # Convert some Aces from 11 -> 1 if needed
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    return total
"""

def hand_value(cards) -> int:
    total=0
    aces = 0
    for card in cards:
        total += VALUES[card[:-1]]
        if card[:-1] == 'A':
            aces+=1

    # Convert some Aces from 11 -> 1 if needed
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    return total

# -----------------------
# Dealer Logic
# -----------------------




# -----------------------
# Utility Function (abstract payoff)
# -----------------------
def utility(agent_hand, dealer_hand) -> int:
    """
    Returns:
      +1 if agent 'wins'
       0 if tie
      -1 if agent 'loses'
    """
    av = hand_value(agent_hand)
    dv = hand_value(dealer_hand)

    if av > 21:
        return -1
    if dv > 21:
        return 1
    if av > dv:
        return 1
    if av < dv:
        return -1
    return 0
