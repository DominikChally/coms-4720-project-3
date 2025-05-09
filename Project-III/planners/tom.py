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

def get_legal_actions(state):

	return


class State:
	def __init__(self, grid, current, pursued, pursuer):
		self.grid = grid
		self.current = current
		self.pursed = pursued
		self.pursuer = pursuer


class Node:
	def __init__(self, state, done, parent, observation, action_index):
		
		#child nodes
		self.children = {}
            
        #total rewards from MCTS exploration
		self.T = 0
            
		#visit count
		self.N = 0
        #environment
		self.state = state
		self.observation = observation
		self.done = done
		self.parent = parent
		self.action_index = action_index
            
	def getUCBscore(self, c=2):
		if self.N == 0:
			return float('inf')

		top_node = self
		if top_node.pare:
			top_node = top_node.parent
		
		return (self.T / self.N) + c * math.sqrt(math.log(top_node.N) / self.N)
      
	def best_child(self, c_val):
		res = max(self.children.items(),
					key=lambda item: self.children[item].getUCBscore(c_val))[1]

		return res
	
	def is_fully_expanded(self):
		return len(self.children) == len(get_legal_actions(self.state))

	def expand(self):
		
		return
	
	#simulation phase
	def rollout(self):
		return
	
	#update the values accordingly
	def backpropogate(self, result, root_player):

		return


def mcts_search(state, player, simulations = 1000):
	directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                  	  		   [-1, -1], [-1, 1], [1, -1], [1, 1]]) 
	return directions[np.random.choice(9)]
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


		return mcts_search(State(grid, current, pursued, pursuer), 0)


