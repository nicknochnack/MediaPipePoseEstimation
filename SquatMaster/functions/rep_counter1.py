def rep_counter(angle,upper_limit,lower_limit,min_rep_count,min_rep_time,time,stage,rep_count,last_event_time,real_counter):

    """
    Function use description:
    To see help on how to use the rep_counter function, type: help(rep_counter) after importing it
    ...
    """


    # Above the upper angle limit, register the "hold up" stage
    if angle >= upper_limit:
        stage = 'hold up'
    # Below the upper limit and after the "hold up" stage, register "down" stage
    if angle < upper_limit and stage == 'hold up':
        stage = 'down'
    # Below the lower limit and after the "down" stage, register the "hold down" stage
    if angle < lower_limit and stage =='down':
        stage = 'hold down'
    # Above the lower limit and after the "hold down" stage, register the "up" stage and count the rep
    if angle > lower_limit and stage == 'hold down':
        stage = 'up'
        rep_count +=1
        last_event_time = time
        print(last_event_time)

    # Register the real rep count only for sets with more than the min rep count
    if rep_count >= min_rep_count:
        real_counter = rep_count
    # Restart the counter when the reps do not repeat in less than the min rep time
    if time - last_event_time > min_rep_time:
        rep_count = 0

    return stage,rep_count,last_event_time,real_counter
