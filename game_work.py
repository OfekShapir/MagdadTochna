from game_env import Deck, dealer_play, utility, hand_value
from bj_player import Agent


def pre_deal_environment(deck, agent, num_other_players=3, cards_per_player=2):
    """
    Simulate other players getting cards AND reveal dealer upcard.

    - Draw num_other_players * cards_per_player cards (others)
    - Then draw 1 dealer upcard
    - Update count for all seen cards
    - Return the dealer upcard
    """
    # Other players' visible cards (we assume agent sees them)
    for _ in range(num_other_players * cards_per_player):
        if deck.size() == 0:
            break
        card = deck.draw()
        agent.update_count(card)

    # Dealer upcard
    if deck.size() == 0:
        raise RuntimeError("Deck empty before dealer upcard")

    dealer_upcard = deck.draw()
    agent.update_count(dealer_upcard)

    return dealer_upcard


def play_round(agent, deck):
    # 1. Environment + dealer upcard
    dealer_upcard = pre_deal_environment(deck, agent)

    # 2. Decide if we join this round or skip
    if not agent.should_enter(dealer_upcard):
        # We still need dealer to complete hand (for "other players")
        dealer_hand = dealer_play(deck, dealer_upcard)
        # We see all dealer cards, so update count
        for c in dealer_hand[1:]:  # skip upcard, already counted
            agent.update_count(c)
        return "skipped"

    # 3. Allocate units (abstract weight)
    units = agent.allocate_units()

    # 4. Our hand (this is what you'd get from vision in your project)
    hand = [deck.draw(), deck.draw()]
    for c in hand:
        agent.update_count(c)

    # 5. Agent’s hit/stand loop
    while True:
        action = agent.decide_action(hand, deck, dealer_upcard)
        if action == "stand":
            break

        card = deck.draw()
        hand.append(card)
        agent.update_count(card)

        if hand_value(hand) > 21:
            break

    # 6. Dealer finishes hand
    dealer_hand = dealer_play(deck, dealer_upcard)
    for c in dealer_hand[1:]:  # upcard already counted
        agent.update_count(c)

    # 7. Compute abstract result & update resources
    result = utility(hand, dealer_hand)   # -1 / 0 / +1
    agent.resources += result * units     # scale by units

    return "played"


def main():
    deck = Deck()
    agent = Agent()

    num_rounds = 1000

    for round_idx in range(num_rounds):
        # Shuffle when deck low, reset count (new shoe)
        if deck.size() < 20:
            deck = Deck()
            agent.running_count = 0

        outcome = play_round(agent, deck)

        # Only print when we actually played, to avoid spam
        if outcome == "played":
            print(f"Round {round_idx+1:04d}: {outcome}, "
                  f"count={agent.running_count}, "
                  f"resources={agent.resources:.2f}")


if __name__ == "__main__":
    main()
