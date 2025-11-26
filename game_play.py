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
        print(f"Bot wants to choose: {'Hit' if move == 'h' else 'Stand'}, Confirm:")
        input()
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
