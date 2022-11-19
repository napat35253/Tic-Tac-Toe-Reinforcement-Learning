# %%
import numpy as np
import copy
from matplotlib import pyplot as plt

# %%
class Board:
    def __init__(self):
        self.state = np.zeros([3,3])
    
    def update_board(self,move,player):
        self.state[move[0],move[1]] = player
        return self.state

    def game_status(self):
        players = np.unique(self.state)
        players = np.delete(players,np.where(players == 0))
        # check winner player
        for player in players:

            # horizontal 
            for row in self.state:
                if np.all(row == player):
                    return player

            # vertical
            for col in self.state.T:
                if np.all(col == player):
                    return player
            
            # diagonal
            if np.all(self.state.diagonal() == player):
                return player

            if np.all(np.fliplr(self.state).diagonal() == player):
                return player
        
        if (np.all(self.state != 0)):
            return "draw"

        return 0



# %%
class Player:

    def __init__(self,name):
        self.name = name
        self.q_table = {}

    def get_state_string(self,state):
        return str(copy.copy(state).reshape(9))

    def get_q_table(self):
        return self.q_table

    def create_new_q_values(self,state,state_string):
        temp_q_values = np.zeros([3,3])
        it = np.nditer(state, flags=['multi_index'])
        for i in it:
            if i != 0 :
                temp_q_values[it.multi_index] = np.nan

        self.q_table[state_string] = temp_q_values
        

    def get_q_values(self,state):
        state_string = self.get_state_string(state)

        if state_string not in self.q_table:
            self.create_new_q_values(state,state_string)      

        return self.q_table[state_string]

    
    def update_q_table(self,state,next_state,action_idx,gamma,epsilon,alpha,reward):

        state_string = self.get_state_string(state)

        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        
        max_q_values = np.nanmax(next_q_values) 

        if np.isnan(max_q_values):
            max_q_values = 0

        q_values[action_idx] += alpha*(reward + gamma * max_q_values - q_values[action_idx])
        self.q_table[state_string] = q_values
        
        return 

    def moves(self,state,epsilon):
        q_values = self.get_q_values(state)
        r = np.random.rand()

        if np.all(np.isnan(q_values)):
            return state

        if r < (1-epsilon) :
            max_q_idx = np.where(q_values == np.nanmax(q_values))
            ran_a_idx = np.random.randint(len(max_q_idx[0]))
            action = (max_q_idx[0][ran_a_idx], max_q_idx[1][ran_a_idx])
        else :
            non_nan_idxs = np.argwhere(~np.isnan(q_values)) 
            r_idx = np.random.randint(non_nan_idxs.shape[0]) 
            action = non_nan_idxs[r_idx,:]
            action = tuple(action)
            
        return action

    def savetxt_compact(self,fname, x, fmt="%.6g", delimiter=','):
        with open(fname, 'w') as fh:
            for row in x:
                line = delimiter.join("0" if value == 0  else fmt % value for value in row)
                fh.write(line + '\n')

    def export_q_table_csv(self,filename):
        
        q_table_list = np.array(list(self.q_table.values()))
        q_table_list = np.concatenate(q_table_list, axis=1)
        merge_board = []

        for key in self.q_table.keys():
            array_result = np.fromstring(key[1:-1],dtype=int, sep='. ').reshape([3,3])
            merge_board.append(array_result)

        merge_board = np.concatenate(merge_board, axis=1)
        final = np.concatenate((merge_board, q_table_list), axis=0)

        self.savetxt_compact(filename, final, fmt='%.6f')

# %%
alpha = 0.1
gamma = 1
epsilon = 1
decay_rate = 0.9999
start_decay = 100
n_epoch = 10**5
count_interval = 100


player1 = Player(1)
player2 = Player(-1)

result_list_count = [0,0,0] #draw,1,2
result_list = [[],[],[]]

reward = {
    'win':1,
    'lost':-1,
    'draw':0,
    'default':0
}

for epoch in range(n_epoch):

    game_finished = False

    board = Board()

    # p1 moves
    state1 = copy.copy(board.state)
    action1 = player1.moves(state1,epsilon)
    board.update_board(action1,player1.name)

    # p2 moves
    state2 = copy.copy(board.state)
    action2 = player2.moves(board.state,epsilon)
    board.update_board(action2,player2.name)

    # update q1
    player1.update_q_table(state1,board.state,action1,gamma,epsilon,alpha,reward['default'])

    while game_finished == False:

        # p1 moves
        state1 = copy.copy(board.state)
        action1 = player1.moves(board.state,epsilon)
        board.update_board(action1,player1.name)

        if board.game_status() != 0:
            result = board.game_status()
            game_finished == True
            break
        else:
            # update q2
            player2.update_q_table(state2,board.state,action2,gamma,epsilon,alpha,reward['default'])

        
        # p2 moves
        state2 = copy.copy(board.state)
        action2 = player2.moves(board.state,epsilon)
        board.update_board(action2,player2.name)

        if board.game_status() != 0:
            result = board.game_status()
            game_finished == True
            break
        else:
            # update q1
            player1.update_q_table(state1,board.state,action1,gamma,epsilon,alpha,reward['default'])

    match result:
        case "draw":
            player1.update_q_table(state1,board.state,action1,gamma,epsilon,alpha,reward['draw'])
            player2.update_q_table(state2,board.state,action2,gamma,epsilon,alpha,reward['draw'])
            result_list_count[0] += 1
        case player1.name:
            player1.update_q_table(state1,board.state,action1,gamma,epsilon,alpha,reward['win'])
            player2.update_q_table(state2,board.state,action2,gamma,epsilon,alpha,reward['lost'])
            result_list_count[1] += 1
        case player2.name:
            player2.update_q_table(state2,board.state,action2,gamma,epsilon,alpha,reward['win'])
            player1.update_q_table(state1,board.state,action1,gamma,epsilon,alpha,reward['lost'])
            result_list_count[2] += 1
            
    epsilon *= decay_rate
    
    # if epoch > start_decay:
    #     if epoch%50 == 0:
    #         epsilon *= decay_rate

    if epoch % 1000 == 0: 
        print('Epoch : {} , Player 1 : {} win , Player 2 : {} win , tie : {} '.format(epoch,result_list_count[1]/count_interval,result_list_count[2]/count_interval,result_list_count[0]/count_interval))


    if epoch % count_interval == 0 and epoch != 0: 
        result_list[0].append(result_list_count[0]) 
        result_list[1].append(result_list_count[1]) 
        result_list[2].append(result_list_count[2]) 
        result_list_count=[0,0,0]

# %%
p1_stat = [x / count_interval for x in result_list[1]] 
p2_stat = [x / count_interval for x in result_list[2]] 
tie_stat = [x / count_interval for x in result_list[0]]

plt.plot(range(len(p1_stat)), p1_stat, label='player 1 win',color="red")
plt.plot(range(len(p2_stat)), p2_stat, label='player 2 win',color="blue")
plt.plot(range(len(tie_stat)), tie_stat, label='tie',color="green")
plt.legend(loc="upper left")
plt.xlabel("epochx200")
plt.ylabel("Result [%]")
plt.title("Reinforcement Learning TIC TAC TOE")
plt.savefig('rl_plot.png', format='png', dpi=1200)

# %%
player1.export_q_table_csv('player1.csv')
player2.export_q_table_csv('player2.csv')


