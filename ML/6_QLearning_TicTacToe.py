"""
Practical 6 : Reinforcement Learning - Tic-Tac-Toe with Q-Learning
==================================================================
INSTRUCTIONS
 1. Install:  pip install matplotlib
 2. Dataset : none - the agent learns by playing itself (self-play).
 3. Run     : python 6_QLearning_TicTacToe.py            (add --play to play against the agent)
 4. Steps   : a. set up environment (board = tuple of 9 cells, 0 empty / 1 X / 2 O)
              b. define the game (legal moves, winner check, reward)
              c. build the RL model (Q-table, epsilon-greedy, Bellman update)
              d. train via self-play
              e. test vs a random player (and optionally vs you)
"""

# --- Step 1: Import the libraries ---
import random, sys
from collections import defaultdict
import matplotlib.pyplot as plt

# --- Step 2: Environment - the board and the rules of the game ---
# Board = tuple of 9 cells (0 empty, 1 X, 2 O); a tuple can be a dict key (a "state").
# The 8 winning lines: 3 rows, 3 columns, 2 diagonals.
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

# Return the winner (1 or 2), 0 if the game goes on, -1 if it is a draw.
def winner(b):
    for i, j, k in LINES:
        if b[i] and b[i] == b[j] == b[k]:
            return b[i]
    return 0 if 0 in b else -1

# Legal moves = empty cells.
def moves(b):       return [i for i in range(9) if b[i] == 0]
# New board with player p at cell a (tuples cannot be edited in place).
def play(b, a, p):  return b[:a] + (p,) + b[a+1:]
# Swap X and O so the mover always sees itself as 1; one Q-table then serves both sides.
def flip(b):        return tuple((0, 2, 1)[c] for c in b)

# --- Step 3: Q-learning agent - the Q-table, epsilon-greedy choice ---
# Q[state][action] = value of playing that cell; unseen states start at zero.
Q = defaultdict(lambda: [0.0] * 9)
# ALPHA = learning rate, GAMMA = how much future reward counts.
ALPHA, GAMMA = 0.3, 0.95

# Best Q-value among legal moves only.
def best_q(s):  return max(Q[s][m] for m in moves(s))

# Epsilon-greedy: explore a random move with chance eps, else play the best known move.
def choose(s, eps):
    mv = moves(s)
    if random.random() < eps:
        return random.choice(mv)
    top = best_q(s)
    # Random tie-break so the agent does not always pick the lowest cell.
    return random.choice([m for m in mv if Q[s][m] == top])

# --- Step 4: Training by self-play (one Q-table plays both sides) ---
# Play many games against itself, learning from each one.
def train(episodes=30000, eps_start=1.0, eps_end=0.05):
    for ep in range(episodes):
        # Exploration falls from 1.0 to 0.05 over the first 80% of games.
        eps = max(eps_end, eps_start * (1 - ep / (episodes * 0.8)))
        # Fresh board, X first; hist remembers every (state, action) of this game.
        b, player, hist = (0,) * 9, 1, []
        while True:
            # The mover always sees itself as 1, so flip the board when O moves.
            s = b if player == 1 else flip(b)
            a = choose(s, eps)
            hist.append((s, a))
            b = play(b, a, player)
            w = winner(b)
            if w != 0:
                break
            player = 3 - player
        # Bellman backup, newest move first. Next state belongs to the opponent -> negate.
        # Q(s,a) moves a little (ALPHA) toward: reward now + discounted best value next.
        won = w != -1
        # Go backwards so the final reward travels back through the game.
        for i in range(len(hist) - 1, -1, -1):
            s_i, a_i = hist[i]
            if i == len(hist) - 1:
                # My final move: +1 if it won, +0.5 for a draw.
                target = 1.0 if won else 0.5
            elif i == len(hist) - 2:
                # Opponent's final move: bad for me if they won, +0.5 for a draw.
                target = -1.0 if won else 0.5
            else:
                # Earlier move: the opponent's gain is my loss, so negate.
                target = -GAMMA * best_q(hist[i + 1][0])
            Q[s_i][a_i] += ALPHA * (target - Q[s_i][a_i])

# --- Step 5: Testing - agent against a random player ---
# Play many games vs a random player and count wins, draws and losses.
def agent_vs_random(games=1000):
    res = {"win": 0, "draw": 0, "loss": 0}
    for g in range(games):
        # Agent alternates between X and O so both sides get tested.
        ag = 1 if g % 2 == 0 else 2
        b, p = (0,) * 9, 1
        while True:
            # Agent plays greedy (eps=0); the opponent plays a random legal move.
            a = choose(b if p == 1 else flip(b), 0) if p == ag else random.choice(moves(b))
            b = play(b, a, p); w = winner(b)
            if w != 0:
                res["draw" if w == -1 else "win" if w == ag else "loss"] += 1
                break
            p = 3 - p
    return res

# Print the board as a 3 x 3 grid.
def show(b):
    sym = {0: ".", 1: "X", 2: "O"}
    for r in range(3):
        print(" ".join(sym[b[3 * r + c]] for c in range(3)))

# Let a human (X, first) play against the trained agent (O).
def human_game():
    b, p = (0,) * 9, 1
    while True:
        show(b)
        if p == 1:
            # Ask for a cell; re-ask on bad input or a taken cell.
            try:
                a = int(input("Your move (cell 0-8): "))
            except ValueError:
                continue
            if a not in moves(b):
                print("Invalid move"); continue
        else:
            # Agent plays its best known move (eps=0).
            a = choose(flip(b), 0)
        b = play(b, a, p); w = winner(b)
        if w != 0:
            show(b); print("Draw" if w == -1 else "You win" if w == 1 else "Agent wins"); return
        p = 3 - p

# --- Step 6: Run everything ---
if __name__ == "__main__":
    # Fixed seed so results repeat on every run.
    random.seed(0)
    # Baseline: an untrained agent should be about even with random.
    print("Untrained agent vs random:", agent_vs_random(500))
    train()
    print("States learned:", len(Q))
    # Test the trained agent and chart win/draw/loss.
    r = agent_vs_random(1000)
    print("Trained agent vs random  :", r)
    plt.bar(list(r.keys()), list(r.values()), color=["g", "gray", "r"])
    plt.title("Q-agent vs random player (1000 games)"); plt.show()
    # Optional: play against the agent yourself.
    if "--play" in sys.argv:
        human_game()

'''
================================================================================
VIVA AND THEORY NOTES - PRACTICAL 6 : Q-LEARNING FOR TIC-TAC-TOE
================================================================================

CODE EXPLANATION
----------------
1. Imports: random (exploration, random opponent), sys (read the --play flag),
   collections.defaultdict (the Q-table), matplotlib (result bar chart).
2. Environment: the board is a tuple of 9 cells (0 empty, 1 X, 2 O). LINES holds the 8
   winning lines. Helper functions:
   - winner(b): returns 1 or 2 if that player has three in a row, -1 for a draw, 0 if the
     game is still running.
   - moves(b): list of legal actions (empty cells).
   - play(b, a, p): returns a new board with player p at cell a.
   - flip(b): swaps X and O so the mover always sees itself as 1. Because of this one
     Q-table learns to play both X and O.
3. Agent: Q = defaultdict(lambda: [0.0]*9) stores 9 action values per state. ALPHA = 0.3,
   GAMMA = 0.95. best_q(s) gives the best value among legal moves. choose(s, eps) is
   epsilon-greedy: random move with probability eps, else best known move with a random
   tie-break.
4. Training (train): 30000 games of self-play. eps decays linearly from 1.0 to 0.05 over the
   first 80 percent of games. Every (state, action) is stored in hist. After the game the
   Bellman backup runs from the last move to the first:
   - last move: target = +1 if it won, +0.5 if the game was a draw;
   - second-last move (opponent ended the game): target = -1 if opponent won, +0.5 if draw;
   - earlier moves: target = -GAMMA * best_q(next state of the opponent);
   - update: Q[s][a] += ALPHA * (target - Q[s][a]).
5. Testing (agent_vs_random): 1000 games, the agent (eps = 0) alternates between X and O
   against a random player; counts win, draw and loss.
6. Main block: seed 0, test the untrained agent (500 games), train, print number of states,
   test again (1000 games), draw the bar chart, and optionally call human_game() if --play
   is given. show() prints the board; human_game() lets a person play X against the agent.

EXPECTED OUTPUT (real run)
--------------------------
   Untrained agent vs random: {'win': 210, 'draw': 62, 'loss': 228}   (500 games, about even)
   States learned: 4520
   Trained agent vs random  : {'win': 909, 'draw': 89, 'loss': 2}     (1000 games)
   i.e. about 91 percent wins, 9 percent draws and 0.2 percent losses after training.
   (Exact numbers depend on the Python random generator; random.seed(0) makes them repeat.)

DETAILED THEORY NOTES
---------------------
1. WHAT IS REINFORCEMENT LEARNING (RL) AND Q-LEARNING?
   - RL: a third kind of machine learning (besides supervised and unsupervised). An AGENT
     takes ACTIONS in an ENVIRONMENT, receives REWARDS, and learns a POLICY that maximises
     the total reward over time. It learns by trial and error, not from labelled data.
   - Q-learning: a model-free, OFF-POLICY, value-based RL algorithm. It learns Q(s, a) = the
     expected total future reward of taking action a in state s and then acting well.
   - Intuition: the Q-table is the agent's memory of "how good was this move from this
     board". Good moves get higher numbers over many games, so later the agent simply picks
     the highest number.
   - Bellman update (maths in words):
         Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
     The new estimate moves a small step (alpha) towards "reward I got now plus the
     discounted best value I can get from the next state". The part in brackets is called
     the TD (temporal difference) error.
   - Two-player twist (used in this code): after my move the next state belongs to the
     opponent, and their gain is my loss. So the next-state value is NEGATED:
         target = - gamma * best_q(opponent's state).
     This is the "negamax" idea. Rewards only appear at the end: +1 win, +0.5 draw, -1 loss.

2. TYPES AND VARIANTS
   - Q-learning (off-policy): updates using the best next action, whatever was actually
     played.
   - SARSA (on-policy): updates using the next action actually taken, so it learns the value
     of the policy it follows (including exploration); usually safer/more cautious.
   - Deep Q-Network (DQN): replaces the Q-table with a neural network so it can handle huge
     or continuous state spaces (for example Atari games from pixels).
   - Double Q-learning: two tables reduce over-estimation of values.
   - Policy-gradient / Actor-Critic (REINFORCE, A2C, PPO): learn the policy directly.
   - Monte Carlo and TD(0) methods: Monte Carlo waits for the end of the episode; TD updates
     step by step. Our backup is a mix: it runs after the episode but bootstraps from the
     Q-table.

3. KEY CONCEPTS AND TERMS
   - Agent : the learner (our Q-table player).
   - Environment : the game (board and rules).
   - State (s) : the board position as seen by the player about to move (tuple of 9).
   - Action (a) : choosing an empty cell, 0 to 8.
   - Reward (r) : number given for the result: +1 win, +0.5 draw, -1 loss, nothing otherwise.
   - Policy : the rule used to pick actions (here epsilon-greedy while learning, greedy when
     testing).
   - Q-value / Q-table : estimated usefulness of each (state, action); a dictionary mapping
     state to a list of 9 numbers.
   - Episode : one complete game from empty board to the end.
   - alpha = 0.3 (learning rate) : how much each update changes the old value. 0 = never
     learns, 1 = completely overwrites. Too high is unstable, too low is slow.
   - gamma = 0.95 (discount factor) : weight of future rewards versus immediate ones. Near
     1 = far-sighted, near 0 = short-sighted. It also makes quicker wins slightly better
     than slow ones.
   - epsilon (exploration rate) : chance to play a random move. Here it decays from 1.0 to
     0.05 over the first 24000 of 30000 games: explore a lot early, exploit later. In
     testing eps = 0 (no exploration).
   - Exploration vs exploitation : try new things to find better moves versus use what is
     already known to win now. A balance is needed.
   - Greedy / tie-break : choose the highest Q-value; if several are equal pick one at
     random so the agent does not always favour cell 0.
   - Self-play : the agent plays both sides, so it needs no teacher and the opponent gets
     stronger as the agent does.
   - Board flip : renaming pieces so the mover is always "1"; lets ONE table play both
     X and O (every state is shown from the mover's view). It does not reduce the number of states.
   - Off-policy : the update uses the best action (max), not necessarily the one chosen by
     the exploring behaviour policy.
   - Model-free : the agent does not need to be told the game's transition rules in advance;
     it learns from experience.
   - Bellman backup (backward pass): processing the game's moves from last to first so that
     the final reward flows backwards to the earlier moves.

4. EVALUATION METRICS
   - Win / draw / loss counts and rates over many games against a random player. For
     Tic-Tac-Toe good play never loses, so the key number is the loss count (2 of 1000 here).
   - Number of states learned (len(Q) = 4520) shows how much of the state space was visited.
     The full game has far fewer legal positions than 3^9 = 19683; 4520 = every non-terminal
     legal position, so the table covers the whole game (flip does not change this count).
   - Confusion matrix, precision, recall, F1, silhouette and WCSS are for supervised
     classification and clustering; they do NOT apply to RL. RL is judged by total reward,
     win rate, or performance against a known opponent.
   - Another common check (not coded): average reward per episode or against a perfect
     minimax player (the trained agent should draw).

5. TRAIN/TEST SPLIT
   - Not used. RL has no fixed dataset: the agent creates its own data by playing. Instead
     "testing" means playing separate games with exploration turned off (eps = 0).
   - random.seed(0) fixes the random numbers so the whole run (training, opponent moves and
     exploration) is repeatable.

6. THE ENVIRONMENT (INSTEAD OF A DATASET)
   - Tic-Tac-Toe: 3 x 3 board, players X (first) and O alternate, three in a row wins, full
     board without a line is a draw.
   - State = tuple of 9 values {0, 1, 2}; 9 possible actions (only empty cells are legal).
   - Training data: 30000 self-play episodes generated on the fly.
   - Small, discrete and fully observable, so a table is enough (about 4.5 thousand states
     stored here).

7. ADVANTAGES
   - No labelled data or human teacher needed; learns from interaction.
   - Simple, easy to understand and guaranteed to converge for small tabular problems when
     every state-action pair is visited enough and alpha is suitably reduced.
   - Off-policy: can learn the best policy while exploring.

8. DISADVANTAGES
   - Table grows with the number of states: state explosion (chess, Go, images are
     impossible) - solved by function approximation such as DQN.
   - Needs many episodes; slow for sparse rewards.
   - Sensitive to alpha, gamma and the epsilon schedule.
   - Cannot generalise: a state never visited has no knowledge, even if similar to a known
     one.
   - Self-play with a shared table can give a policy that is only as good as what it explored.

9. REAL-WORLD USES
   - Game playing (Backgammon, Atari with DQN, board games with AlphaZero-style methods).
   - Robotics control and navigation.
   - Recommendation systems and ad placement.
   - Traffic-light and energy/resource management.
   - Trading strategies and inventory control.
   - Autonomous driving components.

10. COMPARISON WITH RELATED METHODS
   - Q-learning vs SARSA : Q-learning is off-policy (uses max of next state), SARSA is
     on-policy (uses the action actually taken). Q-learning learns the optimal policy more
     directly; SARSA is more conservative when exploring is risky.
   - Q-learning vs DQN : tabular Q-learning stores values in a table and suits small
     discrete spaces; DQN approximates Q with a neural network (plus experience replay and a
     target network) and handles huge or continuous states.
   - Q-learning vs supervised learning : supervised learning needs labelled input-output
     pairs; RL learns from delayed rewards through trial and error.
   - Q-learning vs minimax : minimax searches the whole game tree and needs a full model of
     the game; Q-learning learns from play and needs no model.
   - Q-learning vs Monte Carlo : Monte Carlo waits for the end of every episode and uses
     the full return; Q-learning bootstraps from its own next-state estimate.

LIBRARIES AND FUNCTIONS USED
----------------------------
- random : random.random (a number in 0..1 to compare with eps), random.choice (pick a
  random move / break ties / random opponent), random.seed (repeatable run).
- sys : sys.argv - checks whether "--play" was given on the command line.
- collections.defaultdict : the Q-table, auto-creates [0.0]*9 for unseen states.
- matplotlib.pyplot (plt) : bar (win/draw/loss chart), title, show.
- Built-ins : range, max, len, int, input, print, list, tuple, dict.
- Own functions : winner, moves, play, flip (game rules); best_q, choose (agent);
  train (learning); agent_vs_random (testing); show, human_game (play with a person).

DETAILED VIVA Q&A
-----------------
Q1. What is reinforcement learning?
A1. It is a type of machine learning where an agent interacts with an environment, receives
    rewards or penalties for its actions, and learns a policy that maximises the total
    reward. There is no labelled data; learning is by trial and error.

Q2. What is Q-learning and what is the Q-table?
A2. Q-learning is a model-free, off-policy algorithm that learns the value Q(s, a) of taking
    action a in state s. The Q-table stores one such value for every state-action pair; once
    learned, the agent just picks the action with the highest value.

Q3. Explain the Bellman update in simple words.
A3. New Q = old Q + alpha * (target - old Q), where target = reward + gamma * best value of
    the next state. In words: move the old estimate a little towards "what I got now plus
    the best I expect later". In our two-player game the next-state value is negated because
    the next state belongs to the opponent.

Q4. What are alpha and gamma?
A4. Alpha (0.3) is the learning rate: how strongly a new experience changes the old Q-value.
    Gamma (0.95) is the discount factor: how much future rewards count compared with
    immediate ones. A gamma near 1 makes the agent plan for the long term.

Q5. What is the exploration vs exploitation trade-off and how does epsilon-greedy handle it?
A5. The agent must try new moves (explore) to find better ones, but also use its current
    knowledge (exploit) to win. Epsilon-greedy plays a random move with probability eps and
    the best known move otherwise. Here eps starts at 1.0 and decays to 0.05.

Q6. Q-learning vs SARSA?
A6. Q-learning is off-policy: it updates with the maximum Q-value of the next state,
    regardless of the move actually played. SARSA is on-policy: it uses the next action
    actually chosen, so it also accounts for exploration. SARSA is usually more cautious.

Q7. What are the limits of tabular Q-learning and how does DQN help?
A7. The table must have an entry for every state, so it explodes for big or continuous
    spaces and cannot generalise to unseen states. DQN replaces the table with a neural
    network that estimates Q-values, so similar states share knowledge.

Q8. What is the reward design in this practical?
A8. +1 for a win, +0.5 for a draw, -1 for a loss, and no reward during the game. Draw is
    positive because it is much better than losing, and against good play it is the best
    possible result.

Q9. What is self-play and why is it useful?
A9. The agent plays both X and O, using one Q-table. It needs no human opponent or data, and
    as it improves its opponent improves too, producing steadily harder practice.

Q10. Why is Q-learning called model-free?
A10. It does not need to know the transition probabilities or rules in advance; it only
     learns from the states, actions and rewards it experiences. (We coded the rules only to
     generate the experience.)

PRACTICAL AND CODING VIVA QUESTIONS
-----------------------------------
Q1. Why is the board stored as a tuple and not a list?
A1. A tuple is immutable and hashable, so it can be used as a key in the Q dictionary. A
    list cannot. That is also why play() builds a new tuple using slicing.

Q2. What does flip() do and why is it needed?
A2. It swaps 1 and 2 (0 stays 0) using the mapping (0, 2, 1). The player to move always
    sees itself as 1, so one Q-table works for both X and O and the states are shared.

Q3. Why does best_q use only legal moves?
A3. Occupied cells must never be chosen. Looking only at moves(s) stops illegal cells,
    which still have value 0.0, from being picked as the "best" when real moves have
    negative values.

Q4. Explain the three target cases inside train().
A4. Last move: target is +1 for a win or +0.5 for a draw. Second-last move: the opponent
    just ended the game, so the target is -1 if they won or +0.5 for a draw. Earlier moves:
    target = -GAMMA * best_q(next state of the opponent). The Q update then moves Q[s][a]
    towards that target by ALPHA.

Q5. Why do we loop backwards through hist?
A5. The final reward is known only at the end. Going from the last move to the first lets
    each earlier state use the already updated value of the following state, so the reward
    travels back through the game in one pass.

Q6. How does the epsilon schedule work?
A6. eps = max(0.05, 1.0 * (1 - ep / (30000 * 0.8))). It starts at 1.0 (pure random), falls
    linearly, reaches 0.05 at game 24000 and stays at 0.05 for the last 6000 games.

Q7. Why is eps = 0 in agent_vs_random and human_game?
A7. During testing and real play we want the agent to use what it learned (greedy), not make
    random moves, so exploration is turned off with eps = 0.

Q8. Why does the agent alternate between X and O in testing?
A8. Playing first and playing second are different situations. Alternating (ag = 1 if g % 2
    == 0 else 2) checks that the agent is good as both, and the result is fair.

Q9. What does "States learned: 4520" mean?
A9. len(Q) counts how many distinct board positions (from the mover's view) were stored in
    the Q-table during training. These are the positions the agent actually met in 30000
    games.

Q10. Why is the trained agent not shown to win 100 percent?
A10. The opponent in the test is random, and the agent started second in half the games;
     it could still draw or very rarely lose when its table has not seen or fully learned a
     position. Our run gave 909 wins, 89 draws and 2 losses in 1000 games, which is a big
     improvement over the untrained 210 wins and 228 losses in 500 games.

END OF VIVA NOTES
'''
