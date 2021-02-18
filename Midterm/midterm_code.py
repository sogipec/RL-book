from dataclasses import dataclass
from typing import  Tuple,Mapping
from rl.distribution import Constant
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        StateActionMapping,
                                        ActionMapping,FinitePolicy)
from rl.dynamic_programming import (value_iteration_result,value_iteration,
                                    greedy_policy_from_vf)
from grid_maze import maze_grid,SPACE,BLOCK,GOAL
from rl.markov_process import StateReward
import time

@dataclass(frozen=True)
class Coordinate:
    #This is our state class that we call Coordinate
    x: int
    y: int
    
@dataclass(frozen=True)    
class Action:
    #This is our action class
    action: str
    
#We are going to model both formulations using instances
#of FiniteMarkovDecisionProcess
class MazeModel1(FiniteMarkovDecisionProcess[Coordinate,Action]):
    def __init__(
        self,
        grid:dict):
        
        super().__init__(self.get_maze_mapping(grid))

    def get_maze_mapping(self,grid) -> StateActionMapping[Coordinate, Action]:
        mapping: StateActionMapping[Coordinate, Action] = {}
        #To define the StateActionMapping, we need to parse the grid
        #And see which actions are available from each space depending
        #On the surrounding blocks and spaces
        for i in grid.keys():
            x,y = i
            if grid[i] == GOAL:
                mapping[Coordinate(x,y)] = None
            elif grid[i] == SPACE:
                submapping: ActionMapping[Action,StateReward[Coordinate]] = {}
                
                if grid.get((x+1,y)) == SPACE:
                    submapping[Action('RIGHT')] = Constant((Coordinate(x+1,y),0))
                elif grid.get((x+1,y)) == GOAL:
                    submapping[Action('RIGHT')] = Constant((Coordinate(x+1,y),1))
                    
                if grid.get((x-1,y)) == SPACE:
                    submapping[Action('LEFT')] = Constant((Coordinate(x-1,y),0))
                elif grid.get((x-1,y)) == GOAL:
                    submapping[Action('LEFT')] = Constant((Coordinate(x-1,y),1)) 
                    
                if grid.get((x,y+1)) == SPACE:
                    submapping[Action('DOWN')] = Constant((Coordinate(x,y+1),0))
                elif grid.get((x,y+1)) == GOAL:
                    submapping[Action('DOWN')] = Constant((Coordinate(x,y+1),1)) 
                    
                if grid.get((x,y-1)) == SPACE:
                    submapping[Action('UP')] = Constant((Coordinate(x,y-1),0))
                elif grid.get((x,y-1)) == GOAL:
                    submapping[Action('UP')] = Constant((Coordinate(x,y-1),1))
                mapping[Coordinate(x,y)] = submapping
        return mapping

class MazeModel2(FiniteMarkovDecisionProcess[Coordinate,Action]):
    def __init__(
        self,
        grid:dict):
        super().__init__(self.get_maze_mapping(grid))

    def get_maze_mapping(self,grid) -> StateActionMapping[Coordinate, Action]:
        mapping: StateActionMapping[Coordinate, Action] = {}
        for i in grid.keys():
            x,y = i
            if grid[i] == GOAL:
                mapping[Coordinate(x,y)] = None
            elif grid[i] == SPACE:
                submapping: ActionMapping[Action,StateReward[Coordinate]] = {}
                
                if grid.get((x+1,y)) == SPACE:
                    submapping[Action('RIGHT')] = Constant((Coordinate(x+1,y),-1))
                elif grid.get((x+1,y)) == GOAL:
                    submapping[Action('RIGHT')] = Constant((Coordinate(x+1,y),100))
                    
                if grid.get((x-1,y)) == SPACE:
                    submapping[Action('LEFT')] = Constant((Coordinate(x-1,y),-1))
                elif grid.get((x-1,y)) == GOAL:
                    submapping[Action('LEFT')] = Constant((Coordinate(x-1,y),100)) 
                    
                if grid.get((x,y+1)) == SPACE:
                    submapping[Action('DOWN')] = Constant((Coordinate(x,y+1),-1))
                elif grid.get((x,y+1)) == GOAL:
                    submapping[Action('DOWN')] = Constant((Coordinate(x,y+1),100)) 
                    
                if grid.get((x,y-1)) == SPACE:
                    submapping[Action('UP')] = Constant((Coordinate(x,y-1),-1))
                elif grid.get((x,y-1)) == GOAL:
                    submapping[Action('UP')] = Constant((Coordinate(x,y-1),100))
                mapping[Coordinate(x,y)] = submapping
        return mapping
    
if __name__ == '__main__':
    model1 = MazeModel1(maze_grid)
    model2 = MazeModel2(maze_grid)
    TOLERANCE = 1e-5
    #We're using the value_iteration method or rl.dynamic_programming
    def solution(model: FiniteMarkovDecisionProcess[Coordinate,Action],
                    gamma: float
                    )->Tuple[int,Mapping[Coordinate, float],
                             FinitePolicy[Coordinate,Action]]:
        count = 0
        v = value_iteration(model, gamma)
        a = next(v, None)
        while True:
            if a is None:
                break
            b = next(v,None)
            if max(abs(a[s] - b[s]) for s in a) < TOLERANCE:
                break   
            a = b
            count+=1
            
        opt_policy :FinitePolicy[Coordinate,Action] = greedy_policy_from_vf(
            model,
            b,
            gamma
        )
        return count,b,opt_policy
    
    start = time.time()
    count1,opt_vf1,opt_pol1 = solution(model1,0.8)
    print(f"Method 1 took {time.time()-start} to converge")
    start = time.time()
    count2,opt_vf2,opt_pol2 = solution(model2,1)
    print(f"Method 2 took {time.time()-start} to converge")
    print(f"Solution 1 took {count1} iterations to converge")
    print(f"Solution 2 took {count2} iterations to converge")
    print(opt_pol1)
    print(opt_pol2)
    
    #This is a fast solution where we don't track 
    #the number of iterations to converge
    #We're using a built-in function of rl.dynamic_programming here
    start = time.time()
    opt_vf1,opt_pol1 = value_iteration_result(model1,0.8)
    print(f"Method 1 took {time.time()-start} to converge")
    start = time.time()
    opt_vf2,opt_pol2 = value_iteration_result(model2,1)
    print(f"Method 2 took {time.time()-start} to converge")
    