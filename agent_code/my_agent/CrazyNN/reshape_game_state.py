def reshape_game_state(game_state,field_size, vector=True):
    '''
    INPUT: game_state dict, size of field, vector = True
    OUTPUT: all relevant game info, in the shape of [field_size,field_size] or as stacked vector of length 5*field_size**2
    
    Function reshapes all information in game state dictionary and 
    returns info in the form of maps or as unravelled stacked vector
    '''
    #extract all data from coin, bomb and players to maps
    coin_map = np.zeros((field_size,field_size))
    for coin_coord in game_state['coins']:
        coin_map[coin_coord]=1
        
    bomb_map = np.zeros((field_size,field_size))
    for bomb_coord,bomb_time in game_state['bombs']:
        bomb_map[bomb_coord]=bomb_time
        
    player_map = np.zeros((field_size,field_size))
    for name,score,bomb,coord in game_state['others']:
        if bomb:
            player_map[coord]= -1
        else:
            player_map[coord]= -0.5
    name,score,bomb,coord = game_state['self']
    if bomb:
        player_map[coord]= 1
    else:
        player_map[coord]= 0.5
        
    #join all maps
    game_info = np.stack([game_state['field'],game_state['explosion_map'],coin_map,bomb_map,player_map])
    if vector:
        game_info = game_info.reshape(-1)
    
    return game_info