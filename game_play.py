import random

running_count = 0
# Card deck setup
suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
          '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11}


def create_deck():
    deck = [(rank, suit) for suit in suits for rank in ranks]
    random.shuffle(deck)
    parallel_deck = [card[0] for card in deck]
    return deck, parallel_deck


def calculate_hand_value(hand):
    value = sum(values[card[0]] for card in hand)
    aces = sum(card[0] == 'A' for card in hand)
    while value > 21 and aces:
        value -= 10
        aces -= 1
    return value


def print_hand(player_name, hand, hide_first=False):
    if hide_first:
        print(f"{player_name}'s hand: [Hidden], {hand[1][0]} of {hand[1][1]}")
    else:
        cards = ', '.join([f"{r} of {s}" for (r, s) in hand])
        print(f"{player_name}'s hand: {cards}  (value: {calculate_hand_value(hand)})")


# --------------------------------------------------------------
#          BOT LOGIC — EASY TO EDIT THIS BLOCK
# --------------------------------------------------------------
def bot_decision(hand, dealer_upcard_rank):
    """
    The bot sees the dealer's upcard rank (e.g. '6', 'A', 'K').
    """

    value = calculate_hand_value(hand)

    if dealer_upcard_rank in ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']:
        if value < 17:
            return 'h'
        return 's'
    if dealer_upcard_rank in ['2', '3']:
        if value < 13:
            return 'h'
        return 's'

    if dealer_upcard_rank in ['4', '5', '6']:
        if value < 12:
            return 'h'
        return 's'
# --------------------------------------------------------------


def play_round(deck, pdeck):
    if len(deck) < 10:
        print("\nNot enough cards left to start another round. Game over!")
        return None, None, 0

    print("\n=== New Round ===")
    num_players = int(input("Enter number of HUMAN players (bot is Player 1 automatically): "))

    total_players = num_players + 1
    min_cards_needed = (total_players + 1) * 2

    if len(deck) < min_cards_needed:
        print("Not enough cards for all players. Ending game.")
        return None, None, 0

    # Initialize hands
    players = {}

    # Bot Player
    players["Bot Player"] = [deck.pop(), deck.pop()]
    pdeck.pop(); pdeck.pop()

    # Human players
    for i in range(1, num_players + 1):
        players[f"Player {i}"] = [deck.pop(), deck.pop()]
        pdeck.pop(); pdeck.pop()

    # Dealer
    dealer = [deck.pop(), deck.pop()]
    pdeck.pop(); pdeck.pop()

    # SHOW DEALER UPCARD
    print("\n=== Dealer Shows Upcard ===")
    print(f"Dealer's shown card: {dealer[1][0]} of {dealer[1][1]}")
    dealer_upcard_rank = dealer[1][0]

    # --- Bot turn ---
    print("\n--------------------------------")
    print(">>> BOT TURN <<<")
    print_hand("Bot Player", players["Bot Player"])

    while True:
        move = bot_decision(players["Bot Player"], dealer_upcard_rank)
        print(f"Bot chooses: {'Hit' if move == 'h' else 'Stand'}")

        if move == 'h':
            if not deck:
                print("Deck empty! Bot cannot hit.")
                break
            card = deck.pop()
            players["Bot Player"].append(card)
            pdeck.pop()
            print_hand("Bot Player", players["Bot Player"])
            if calculate_hand_value(players["Bot Player"]) > 21:
                print("Bot busts!")
                break
        else:
            break

    # --- Human players ---
    for name, hand in players.items():
        if name == "Bot Player":
            continue

        print("\n--------------------------------")
        while True:
            print_hand(name, hand)
            if calculate_hand_value(hand) > 21:
                print("Bust! Over 21.")
                break

            move = input(f"{name}, Hit or Stand? (h/s): ").strip().lower()
            if move == 'h':
                if not deck:
                    print("Deck is empty! No more cards.")
                    break
                card = deck.pop()
                hand.append(card)
                pdeck.pop()
            else:
                break

    # Dealer turn
    print("\n=== Dealer's Turn ===")
    print_hand("Dealer", dealer)

    while calculate_hand_value(dealer) < 17 and deck:
        card = deck.pop()
        dealer.append(card)
        pdeck.pop()
        print_hand("Dealer", dealer)

    dealer_value = calculate_hand_value(dealer)
    if dealer_value > 21:
        print("Dealer busts!")

    # Results
    print("\n=== Final Results ===")
    print_hand("Dealer", dealer)

    dealer_won = 0  # <--- NEW
    player_won = 0  # <--- NEW

    for name, hand in players.items():
        player_value = calculate_hand_value(hand)
        print_hand(name, hand)

        if player_value > 21:
            print(f"{name} loses (busted).")
            dealer_won += 1
        elif dealer_value > 21 or player_value > dealer_value:
            print(f"{name} wins!")
            player_won += 1
        elif player_value == dealer_value:
            print(f"{name} pushes (tie).")
        else:
            print(f"{name} loses.")
            dealer_won += 1   # dealer beats player

    return deck, pdeck, dealer_won, player_won


def play_blackjack():
    print("=== Blackjack Multi-Round (Bot + Humans) ===")
    deck, pdeck = create_deck()

    rounds = 0
    dealer_total_wins = 0
    player_total_wins = 0

    while True:
        result = play_round(deck, pdeck)

        if result[0] is None:
            print(f"\nCards remaining in deck: {len(deck)}")
            shuffle_choice = input("Shuffle the deck? (y/n): ").strip().lower()
            if shuffle_choice != 'y':
                break
            deck, pdeck = create_deck()
            print(f"\nCards remaining in deck: {len(deck)}")
            break

        deck, pdeck, dealer_won, player_won = result
        rounds += 1
        dealer_total_wins += dealer_won
        player_total_wins += player_won

        print(f"\n=== STATS AFTER ROUND {rounds} ===")
        print(f"Dealer total wins so far: {dealer_total_wins}")
        print(f"Bot Player total wins so far: {player_total_wins}")
        print("=====================================")

        if not deck or len(deck) < 10:
            print(f"\nCards remaining in deck: {len(deck)}")
            shuffle_choice = input("Shuffle the deck? (y/n): ").strip().lower()
            if shuffle_choice != 'y':
                break
            deck, pdeck = create_deck()
            print(f"\nCards remaining in deck: {len(deck)}")

        again = input("\nPlay another round? (y/n): ").strip().lower()
        if again != 'y':
            break
        shuffle_choice = input("Shuffle the deck? (y/n): ").strip().lower()
        if shuffle_choice == 'y':
            deck, pdeck = create_deck()
        print(f"\nCards remaining in deck: {len(deck)}")


    print("\nThanks for playing! Final cards remaining:", len(deck))
    print(f"Total rounds played: {rounds}")
    print(f"Bot Player total wins: {player_total_wins}")
    print(f"Dealer total wins: {dealer_total_wins}")


if __name__ == "__main__":
    play_blackjack()


import random
from collections import Counter, defaultdict

# Card deck setup
suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
          '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11}


def create_deck():
    deck = [(rank, suit) for suit in suits for rank in ranks]
    random.shuffle(deck)
    parallel_deck = [card[0] for card in deck]  # only ranks
    return deck, parallel_deck


def calculate_hand_value(hand):
    value = sum(values[card[0]] for card in hand)
    aces = sum(card[0] == 'A' for card in hand)
    while value > 21 and aces:
        value -= 10
        aces -= 1
    return value


def print_hand(player_name, hand, hide_first=False):
    if hide_first:
        print(f"{player_name}'s hand: [Hidden], {hand[1][0]} of {hand[1][1]}")
    else:
        cards = ', '.join([f"{r} of {s}" for (r, s) in hand])
        print(f"{player_name}'s hand: {cards}  (value: {calculate_hand_value(hand)})")


# =========================
#    EV HELPER FUNCTIONS
# =========================

def get_rank_probs(pdeck):
    """Return probability of each rank based on remaining pdeck.
       If pdeck is empty, assume uniform 1/13 for all ranks."""
    if pdeck:
        c = Counter(pdeck)
        total = len(pdeck)
        return {r: c[r] / total for r in ranks}
    else:
        # Fallback: uniform
        return {r: 1.0 / len(ranks) for r in ranks}


def add_card_to_state(total, usable_aces, rank):
    """Update (total, usable_aces) when drawing a rank."""
    if rank == 'A':
        total += 11
        usable_aces += 1
    else:
        total += values[rank]

    while total > 21 and usable_aces > 0:
        total -= 10
        usable_aces -= 1

    return total, usable_aces


def hand_state_from_hand(hand):
    """Convert a real hand (list of (rank, suit)) to (total, usable_aces)."""
    total = 0
    usable_aces = 0
    for rank, _ in hand:
        if rank == 'A':
            total += 11
            usable_aces += 1
        else:
            total += values[rank]

    while total > 21 and usable_aces > 0:
        total -= 10
        usable_aces -= 1

    return total, usable_aces


def dealer_final_probs_from_state(total, usable_aces, rank_probs, dealer_cache):
    """
    Recursively compute probability distribution over dealer final outcomes
    starting from (total, usable_aces), with dealer hitting on <17 and
    standing on 17+ (including soft 17).
    Outcomes keys: 17,18,19,20,21,'bust'
    """
    key = (total, usable_aces)
    if key in dealer_cache:
        return dealer_cache[key]

    # Dealer stands on 17+
    if total >= 17:
        if total > 21:
            dist = {'bust': 1.0}
        else:
            dist = {total: 1.0}
        dealer_cache[key] = dist
        return dist

    # Dealer must hit
    dist = defaultdict(float)
    for rank, p in rank_probs.items():
        new_total, new_aces = add_card_to_state(total, usable_aces, rank)
        subdist = dealer_final_probs_from_state(new_total, new_aces,
                                                rank_probs, dealer_cache)
        for outcome, p_sub in subdist.items():
            dist[outcome] += p * p_sub

    dealer_cache[key] = dist
    return dist


def dealer_ev_against_player(player_total, rank_probs, dealer_upcard_rank, dealer_cache):
    """
    Compute EV of the dealer vs a fixed player_total, assuming:
    - Dealer upcard is known (dealer_upcard_rank)
    - Hole card is unknown, drawn according to rank_probs
    - Future dealer draws also use rank_probs
    EV is from the player's perspective: +1 win, 0 tie, -1 loss.
    """
    outcome_probs = defaultdict(float)

    # Consider all possible hole cards
    for hole_rank, p_hole in rank_probs.items():
        total = 0
        usable_aces = 0

        # First add upcard
        total, usable_aces = add_card_to_state(total, usable_aces, dealer_upcard_rank)
        # Then add hole card
        total, usable_aces = add_card_to_state(total, usable_aces, hole_rank)

        subdist = dealer_final_probs_from_state(total, usable_aces,
                                                rank_probs, dealer_cache)

        for outcome, p_out in subdist.items():
            outcome_probs[outcome] += p_hole * p_out

    # Now convert outcome distribution into EV vs player_total
    ev = 0.0
    for outcome, p in outcome_probs.items():
        if outcome == 'bust':
            ev += p * (+1)
        elif outcome < player_total:
            ev += p * (+1)
        elif outcome == player_total:
            ev += p * 0
        else:
            ev += p * (-1)

    return ev


def EV_stand_local(player_total, rank_probs, dealer_upcard_rank,
                   dealer_cache, stand_cache):
    """EV if player stands with player_total."""
    if player_total in stand_cache:
        return stand_cache[player_total]

    ev = dealer_ev_against_player(player_total, rank_probs,
                                  dealer_upcard_rank, dealer_cache)
    stand_cache[player_total] = ev
    return ev


def EV_player_optimal(total, usable_aces, rank_probs, dealer_upcard_rank,
                      dealer_cache, stand_cache, player_cache):
    """
    EV of the game for the player from state (total, usable_aces),
    assuming optimal strategy (hit or stand) from here onwards.
    """
    key = (total, usable_aces)
    if key in player_cache:
        return player_cache[key]

    # If already busted
    if total > 21:
        return -1.0

    # EV if we stand now
    ev_stand = EV_stand_local(total, rank_probs, dealer_upcard_rank,
                              dealer_cache, stand_cache)

    # EV if we hit now (and then play optimally after the draw)
    ev_hit = 0.0
    for rank, p in rank_probs.items():
        new_total, new_aces = add_card_to_state(total, usable_aces, rank)
        ev_hit += p * EV_player_optimal(new_total, new_aces, rank_probs,
                                        dealer_upcard_rank, dealer_cache,
                                        stand_cache, player_cache)

    ev_opt = max(ev_stand, ev_hit)
    player_cache[key] = ev_opt
    return ev_opt


# --------------------------------------------------------------
#          BOT LOGIC — EV-BASED DECISION
# --------------------------------------------------------------
def bot_decision(hand, dealer_upcard_rank, pdeck):
    """
    Bot chooses the action with higher expected value:
        if EV(hit) > EV(stand): hit
        else: stand

    Uses:
    - rank probabilities from current pdeck (approx infinite-deck model)
    - full recursion for multiple hits
    - dealer recursion for all possible outcomes
    """
    rank_probs = get_rank_probs(pdeck)
    total, usable_aces = hand_state_from_hand(hand)

    dealer_cache = {}
    stand_cache = {}
    player_cache = {}

    # EV if we stand right now
    ev_stand = EV_stand_local(total, rank_probs, dealer_upcard_rank,
                              dealer_cache, stand_cache)

    # EV if we hit once now, then play optimally after
    ev_hit = 0.0
    for rank, p in rank_probs.items():
        new_total, new_aces = add_card_to_state(total, usable_aces, rank)
        ev_hit += p * EV_player_optimal(new_total, new_aces, rank_probs,
                                        dealer_upcard_rank, dealer_cache,
                                        stand_cache, player_cache)

    print(f"Bot EV(stand) = {ev_stand:.3f}, EV(hit) = {ev_hit:.3f}")

    return 'h' if ev_hit > ev_stand else 's'
# --------------------------------------------------------------


def play_round(deck, pdeck):
    if len(deck) < 10:
        print("\nNot enough cards left to start another round. Game over!")
        return None, None, 0, 0

    print("\n=== New Round ===")
    num_players = int(input("Enter number of HUMAN players (bot is Player 1 automatically): "))

    total_players = num_players + 1
    min_cards_needed = (total_players + 1) * 2

    if len(deck) < min_cards_needed:
        print("Not enough cards for all players. Ending game.")
        return None, None, 0, 0

    # Initialize hands
    players = {}

    # Bot Player
    players["Bot Player"] = [deck.pop(), deck.pop()]
    pdeck.pop(); pdeck.pop()

    # Human players
    for i in range(1, num_players + 1):
        players[f"Player {i}"] = [deck.pop(), deck.pop()]
        pdeck.pop(); pdeck.pop()

    # Dealer
    dealer = [deck.pop(), deck.pop()]
    pdeck.pop(); pdeck.pop()

    # SHOW DEALER UPCARD
    print("\n=== Dealer Shows Upcard ===")
    print(f"Dealer's shown card: {dealer[1][0]} of {dealer[1][1]}")
    dealer_upcard_rank = dealer[1][0]

    # --- Bot turn ---
    print("\n--------------------------------")
    print(">>> BOT TURN <<<")
    print_hand("Bot Player", players["Bot Player"])

    while True:
        move = bot_decision(players["Bot Player"], dealer_upcard_rank, pdeck)
        print(f"Bot chooses: {'Hit' if move == 'h' else 'Stand'}")

        if move == 'h':
            if not deck:
                print("Deck empty! Bot cannot hit.")
                break
            card = deck.pop()
            players["Bot Player"].append(card)
            pdeck.pop()
            print_hand("Bot Player", players["Bot Player"])
            if calculate_hand_value(players["Bot Player"]) > 21:
                print("Bot busts!")
                break
        else:
            break

    # --- Human players ---
    for name, hand in players.items():
        if name == "Bot Player":
            continue

        print("\n--------------------------------")
        while True:
            print_hand(name, hand)
            if calculate_hand_value(hand) > 21:
                print("Bust! Over 21.")
                break

            move = input(f"{name}, Hit or Stand? (h/s): ").strip().lower()
            if move == 'h':
                if not deck:
                    print("Deck is empty! No more cards.")
                    break
                card = deck.pop()
                hand.append(card)
                pdeck.pop()
            else:
                break

    # Dealer turn
    print("\n=== Dealer's Turn ===")
    print_hand("Dealer", dealer)

    while calculate_hand_value(dealer) < 17 and deck:
        card = deck.pop()
        dealer.append(card)
        pdeck.pop()
        print_hand("Dealer", dealer)

    dealer_value = calculate_hand_value(dealer)
    if dealer_value > 21:
        print("Dealer busts!")

    # Results
    print("\n=== Final Results ===")
    print_hand("Dealer", dealer)

    dealer_won = 0
    bot_won = 0

    for name, hand in players.items():
        player_value = calculate_hand_value(hand)
        print_hand(name, hand)

        if name == "Bot Player":
            # Special tracking for bot
            if player_value > 21:
                print(f"{name} loses (busted).")
                dealer_won += 1
            elif dealer_value > 21 or player_value > dealer_value:
                print(f"{name} wins!")
                bot_won += 1
            elif player_value == dealer_value:
                print(f"{name} pushes (tie).")
            else:
                print(f"{name} loses.")
                dealer_won += 1
        else:
            # Human players
            if player_value > 21:
                print(f"{name} loses (busted).")
                dealer_won += 1
            elif dealer_value > 21 or player_value > dealer_value:
                print(f"{name} wins!")
            elif player_value == dealer_value:
                print(f"{name} pushes (tie).")
            else:
                print(f"{name} loses.")
                dealer_won += 1

    print(f"\nCards remaining in deck: {len(deck)}")
    return deck, pdeck, dealer_won, bot_won


def play_blackjack():
    print("=== Blackjack Multi-Round (Bot + Humans) ===")
    deck, pdeck = create_deck()

    rounds = 0
    dealer_total_wins = 0
    bot_total_wins = 0

    while True:
        result = play_round(deck, pdeck)
        if result[0] is None:
            break

        deck, pdeck, dealer_won, bot_won = result
        rounds += 1
        dealer_total_wins += dealer_won
        bot_total_wins += bot_won

        print(f"\n=== STATS AFTER ROUND {rounds} ===")
        print(f"Dealer total wins so far: {dealer_total_wins}")
        print(f"Bot total wins so far: {bot_total_wins}")
        print("=====================================")

        if not deck or len(deck) < 10:
            break

        again = input("\nPlay another round? (y/n): ").strip().lower()
        if again != 'y':
            break

    print("\nThanks for playing! Final cards remaining:", len(deck))
    print(f"Total rounds played: {rounds}")
    print(f"Dealer total wins: {dealer_total_wins}")
    print(f"Bot total wins: {bot_total_wins}")
    return




if __name__ == "__main__":
    play_blackjack()