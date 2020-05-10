python train.py --test_graph_type='erdos' --message_function='mpnn' --message_function_depth=1 --neighbour_state_aggr='sum' --hidden_dim=32 && 
python train.py --test_graph_type='erdos' --message_function='mpnn' --message_function_depth=2 --neighbour_state_aggr='sum' --hidden_dim=16 && 
python train.py --test_graph_type='erdos' --message_function='mpnn' --message_function_depth=1 --neighbour_state_aggr='mean' --hidden_dim=32
