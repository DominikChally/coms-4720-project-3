import math
import numpy as np
from typing import List, Tuple, Optional

'''
Scoring Rules
Each task ends when one of the following occurs:

An agent successfully captures its pursued target without being caught by its own pursuer and without colliding into obstacles.
A collision or stalemate (e.g., no one can catch their target) occurs within a fixed number of steps.
Scoring for each task is as follows:

3 points: Awarded to the winning agent — the first to successfully complete its pursuit objective without being caught or crashing into an obstacle.
0 points: Awarded to the losing agents in a clear win/loss outcome.
1 point each: Awarded to all agents if the task results in a tie (e.g., multiple captures in the same step or a timed-out draw).'''
directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                  	  		   [-1, -1], [-1, 1], [1, -1], [1, 1]]) 
GRID = None


class State:
	def __init__(self, current, pursued, pursuer):
		self.current = current
		self.pursued = pursued
		self.pursuer = pursuer
		self.steps = 0
		#self.max_steps = max_steps
	

	# returns the next move of given state, action
	def move(self, action):
		new_current = [self.current[0] + action[0], self.current[1] + action[1]]
		new_state = State(new_current,
					 self.pursued, self.pursuer)
		return new_state

	def is_game_over(self):

		if np.array_equal(self.current, self.pursued):
			return True
		
		if np.array_equal(self.current, self.pursuer):
			return True
		
		# if self.steps >= self.max_steps:
		# 	return True
		
		return False
	
	def game_result(self):
		# 3 points awarded to winning agent!
		if(np.array_equal(self.current, self.pursuer)):
			return -1.0
		elif (np.array_equal(self.current, self.pursued)):
			return 1.0
		else:
			##draw
			return 0.5
			
	def get_legal_actions(self):
		legal_moves = []
		rows, cols = GRID.shape
        
		for direction in directions:
			new_pos = self.current + direction
            
			# Check if the move is within bounds and not into an obstacle
			if (0 <= new_pos[0] < rows and 
				0 <= new_pos[1] < cols and 
				GRID[new_pos[0], new_pos[1]] == 0):
				legal_moves.append(direction)

		return legal_moves

class Node:
	def __init__(self, state, parent = None, parent_action = None):
		
		#child nodes
		self.children = {}
		self.parent = parent

        #total rewards from MCTS exploration
		self.T = 0
            
		#visit count
		self.N = 0
        #environment
								
		self.state = state
		self.parent_action = parent_action
		self.untried_actions = state.get_legal_actions()

		#self.done = done
            
	def getUCBscore(self, c=1.41):
		if self.N == 0:
			return float('inf')
 		
		exploration = c * math.sqrt(math.log(self.parent.N) / self.N) if self.parent else 0
		exploitation = self.T / self.N
        
		return exploitation + exploration
	
	def best_child(self, c_val=1.41):
		res = max(self.children.items(),
					key=lambda item: item[1].getUCBscore(c_val))[1]

		return res
	
	def is_fully_expanded(self):
		return len(self.untried_actions) == 0

	def is_terminal_node(self):
		return self.state.is_game_over()

	def expand(self):
		# could either do it 'single' (this being just one iteration) or use a loop here to go through them all
		action = self.untried_actions.pop()

		next_state = self.state.move(action)
		child_node = Node(next_state, self, action)
		self.children[tuple(action)] = child_node

		return child_node
	
	#simulation phase
	def rollout(self):
		crnt_rollout_state = self.state
		depth = 0
		max_rollout_depth = 100 		#PROBLEMS HERE, DOESNT ACTUALLY GET TO A TERMINATING STATE
		#loops through all moves until at terminal node
		while not crnt_rollout_state.is_game_over() and depth < max_rollout_depth: # this can either be is_game_over or is_terminal_node
			possible_moves = crnt_rollout_state.get_legal_actions()
			if not possible_moves:
				print("no possible moves")
				break
			
			action = self.rollout_policy(possible_moves, crnt_rollout_state)
			crnt_rollout_state = crnt_rollout_state.move(action)
			depth += 1

		#after simulated gets the result
		return crnt_rollout_state.game_result()
	
	#Randomly select a move for random simulation
	def rollout_policy(self, possible_moves, state):

		#use heuristic to get to choose one in the direction of purserer
		scores = []
		for action in possible_moves:
			new_pos = state.current + action

			# here we can do some heurstics (distance from pursuer and pursued and choose the best
			# action from that)
			# dst_from_pursued = round(np.sqrt((state.pursued[0] - new_pos[0]), 2)
			#  + pow((state.pursued[1] - new_pos[1], 2)), 3)
			# scores.append(dst_from_pursued)
			dst_from_pursued = np.linalg.norm(state.pursued - new_pos)
			scores.append(dst_from_pursued)
		return possible_moves[np.argmin(scores)]
		#return possible_moves[np.random.randint(len(possible_moves))]
	
	#update the values accordingly
	#NEED TO UNDERSTAND RESULT HERE
	def backpropogate(self, result):
		#assign reward here
		
		self.N += 1
		self.T += result #MAYBE MAKE THIS DIFFERENT??

		if self.parent:
			self.parent.backpropogate(result)

def mcts_search(state, simulations = 200):

	root = Node(state)
	for _ in range(simulations):
		node = root

		#selection
		while not node.is_terminal_node() and node.is_fully_expanded():
			node = node.best_child()
			# if node is None:
			# 	break

		#expansion
		if not node.is_terminal_node() and not node.is_fully_expanded():
			node = node.expand()

		#simulation
		result = node.rollout()

		#backpropogation
		node.backpropogate(result)	

	if not root.children:
        # If no legal moves, return no-op
		return np.array([0, 0])

	best_action = root.best_child().parent_action

	# # #change this to highest reward
	# best_action = None
	# best_visits = float('-inf')
	# for action, child in root.children.items():
	# 	if child.N > best_visits:
	# 		best_visits = child.N
	# 		best_action = action

	# Debug information (optional)
	# for action, child in root.children.items():
	# 	print(f"Action {action}: visits={child.N}, value={child.T/child.N if child.N else 0}")
    
	return np.array(best_action) 


# def dfs(grid, start, end):
#     """A DFS example"""
#     rows, cols = len(grid), len(grid[0])
#     stack = [start]
#     visited = set()
#     parent = {start: None}

#     # Consider all 8 possible moves (up, down, left, right, and diagonals)
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
#                   (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal moves

#     while stack:
#         x, y = stack.pop()
#         if (x, y) == end:
#             # Reconstruct the path
#             path = []
#             while (x, y) is not None:
#                 path.append((x, y))
#                 if parent[(x, y)] is None:
#                     break  # Stop at the start node
#                 x, y = parent[(x, y)]
#             return path[::-1]  # Return reversed path

#         if (x, y) in visited:
#             continue
#         visited.add((x, y))

#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and (nx, ny) not in visited:
#                 stack.append((nx, ny))
#                 parent[(nx, ny)] = (x, y)

#     return None

class PlannerAgent:
	
	def __init__(self):
		pass
	
	def plan_action(self, world: np.ndarray, current: np.ndarray, pursued: np.ndarray, pursuer: np.ndarray) -> Optional[np.ndarray]:
		"""
		Computes a action to take from the current position caputure the pursued while evading from the pursuer

		Parameters:
		- world (np.ndarray): A 2D numpy array representing the grid environment.
		- 0 represents a walkable cell.
		- 1 represents an obstacle.
		- current (np.ndarray): The (row, column) coordinates of the current position.
        - pursued (np.ndarray): The (row, column) coordinates of the agent to be pursued.
		- pursuer (np.ndarray): The (row, column) coordinates of the agent to evade from.

		Returns:
		- np.ndarray: one of the 9 actions from 
          					[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                  	  		[-1, -1], [-1, 1], [1, -1], [1, 1]
		"""
		
		directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                  	  		   [-1, -1], [-1, 1], [1, -1], [1, 1]]) 

		#mtcs search returns the best action (one of the directions)
		global GRID
		GRID = world

		initial_state = State(current, pursued, pursuer)

		best_action = mcts_search(initial_state)
		return best_action


