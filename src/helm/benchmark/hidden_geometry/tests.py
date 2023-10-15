hidden_states = raw_completion["hidden_states"]
if kwargs.get("hidden_states", False):
    hidden_states = raw_completion["hidden_states"]
    hidden_states = hidden_states_process(hidden_states)
    #UNIT TEST
    assert type(hidden_states) == dict, f'hidden_states must be a dict instead got {type(hidden_states)}'
    print("Test passed") 
    assert hidden_states.keys() == {"sum", "last"}, \
        f'hidden_states must have keys sum and last  \
        instead it has {hidden_states.keys()}' 
    print("Test passed")
    assert hidden_states["sum"].shape == hidden_states["last"].shape, "hidden_states sum and last must have the same shape"
    print("Test passed")