from GridBoard import *

class Gridworld:

    def __init__(self, width=4, height=4, mode='static', custom_positions=None):
        if width >= 4 and height >= 4:
            self.board = GridBoard(width=width, height=height)
        else:
            print("Minimum board dimensions are 4x4. Initialized to size 4x4.")
            self.board = GridBoard(width=4, height=4)

        self.width = self.board.width
        self.height = self.board.height

        #Add pieces, positions will be updated later
        self.board.addPiece('Player','P',(0,0))
        self.board.addPiece('Goal','+',(1,0))
        self.board.addPiece('Pit','-',(2,0))
        self.board.addPiece('Wall','W',(3,0))

        if custom_positions:
            self.initGridCustom(custom_positions)
        elif mode == 'static':
            self.initGridStatic()
        elif mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()

    def initGridCustom(self, positions):
        if 'Player' in positions: self.board.components['Player'].pos = tuple(positions['Player'])
        if 'Goal' in positions: self.board.components['Goal'].pos = tuple(positions['Goal'])
        if 'Pit' in positions:
            v = positions['Pit']
            # list of positions: e.g. [(1,0), (2,1)] or [[1,0], [2,1]]
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple)):
                self.board.components['Pit'] = [BoardPiece('Pit', '-', tuple(p)) for p in v]
            else:
                self.board.components['Pit'].pos = tuple(v)
        if 'Wall' in positions:
            v = positions['Wall']
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple)):
                self.board.components['Wall'] = [BoardPiece('Wall', 'W', tuple(p)) for p in v]
            else:
                self.board.components['Wall'].pos = tuple(v)


    #Initialize stationary grid, all items are placed deterministically
    def initGridStatic(self):
        #Setup static pieces. For X*Y, we keep the goal/pit/wall fixed near the top-left
        #and the player at the top-right corner.
        self.board.components['Player'].pos = (0, self.width - 1) #Row, Column
        self.board.components['Goal'].pos = (0, 0)
        self.board.components['Pit'].pos = (0, 1)
        self.board.components['Wall'].pos = (1, 1)

    #Check if board is initialized appropriately (no overlapping pieces)
    #also remove impossible-to-win boards
    def validateBoard(self):
        valid = True

        player = self.board.components['Player']
        goal = self.board.components['Goal']
        wall = self.board.components['Wall']
        pit = self.board.components['Pit']

        all_positions = [player.pos, goal.pos]
        if isinstance(wall, list): all_positions.extend([w.pos for w in wall])
        else: all_positions.append(wall.pos)
        
        if isinstance(pit, list): all_positions.extend([p.pos for p in pit])
        else: all_positions.append(pit.pos)

        if len(all_positions) > len(set(all_positions)):
            return False

        corners = [(0,0), (0,self.width-1), (self.height-1,0), (self.height-1,self.width-1)]
        #if player is in corner, can it move? if goal is in corner, is it blocked?
        if player.pos in corners or goal.pos in corners:
            val_move_pl = [self.validateMove('Player', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            val_move_go = [self.validateMove('Goal', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            if 0 not in val_move_pl or 0 not in val_move_go:
                valid = False

        return valid

    #Initialize player in random location, but keep wall, goal and pit stationary
    def initGridPlayer(self):
        self.initGridStatic()
        #place player randomly
        self.board.components['Player'].pos = (np.random.randint(0, self.height), np.random.randint(0, self.width))

        if (not self.validateBoard()):
            self.initGridPlayer()

    #Initialize grid so that goal, pit, wall, player are all randomly placed
    def initGridRand(self):
        self.board.components['Player'].pos = (np.random.randint(0, self.height), np.random.randint(0, self.width))
        self.board.components['Goal'].pos = (np.random.randint(0, self.height), np.random.randint(0, self.width))
        
        num_obstacles = min(self.width, self.height) - 1
        num_pits = max(1, np.random.randint(1, num_obstacles)) if num_obstacles > 1 else 1
        num_walls = num_obstacles - num_pits
        if num_walls < 1 and num_obstacles > 1: num_walls = 1
        elif num_obstacles <= 1: num_walls = 1
        
        pits = []
        for i in range(num_pits):
            pits.append(BoardPiece('Pit', '-', (np.random.randint(0, self.height), np.random.randint(0, self.width))))
        self.board.components['Pit'] = pits
        
        walls = []
        for i in range(num_walls):
            walls.append(BoardPiece('Wall', 'W', (np.random.randint(0, self.height), np.random.randint(0, self.width))))
        self.board.components['Wall'] = walls

        if (not self.validateBoard()):
            self.initGridRand()

    def validateMove(self, piece, addpos=(0,0)):
        outcome = 0 #0 is valid, 1 invalid, 2 lost game
        pit = self.board.components['Pit']
        wall = self.board.components['Wall']
        new_pos = addTuple(self.board.components[piece].pos, addpos)
        
        pit_positions = [p.pos for p in pit] if isinstance(pit, list) else [pit.pos]
        wall_positions = [w.pos for w in wall] if isinstance(wall, list) else [wall.pos]
        
        if new_pos in wall_positions:
            outcome = 1 #block move, player can't move to wall
        elif new_pos[0] > (self.height-1) or new_pos[0] < 0:
            outcome = 1 #outside vertical bounds
        elif new_pos[1] > (self.width-1) or new_pos[1] < 0:
            outcome = 1 #outside horizontal bounds
        elif new_pos in pit_positions:
            outcome = 2

        return outcome

    def makeMove(self, action):
        #need to determine what object (if any) is in the new grid spot the player is moving to
        #actions in {u,d,l,r}
        def checkMove(addpos):
            if self.validateMove('Player', addpos) in [0,2]:
                new_pos = addTuple(self.board.components['Player'].pos, addpos)
                self.board.movePiece('Player', new_pos)

        if action == 'u': #up
            checkMove((-1,0))
        elif action == 'd': #down
            checkMove((1,0))
        elif action == 'l': #left
            checkMove((0,-1))
        elif action == 'r': #right
            checkMove((0,1))
        else:
            pass

    def reward(self):
        pit = self.board.components['Pit']
        pit_positions = [p.pos for p in pit] if isinstance(pit, list) else [pit.pos]
        
        if (self.board.components['Player'].pos in pit_positions):
            return -10
        elif (self.board.components['Player'].pos == self.board.components['Goal'].pos):
            return 10
        else:
            return -1

    def display(self):
        return self.board.render()
