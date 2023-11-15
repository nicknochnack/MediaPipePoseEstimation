

def stage_classifier(angle, exercise_type, image, stage, image_upper,image_middown,image_lower,image_midup, i,j,k,l):
    if exercise_type == 'Squat':
        upper_limit = 160
        lower_limit = 80
        angle_mid = (lower_limit+upper_limit)/2
            
            
        # Above the upper angle limit, register the "hold up" stage
        if angle >= upper_limit:
            stage = 'upper'
            i = 0
            j = 0
            k = 0
            l = 0
        if angle < upper_limit and stage == 'upper':
            if i == 0:
                image_upper = image
                i += 1
                
        # Below the upper limit and after the "hold up" stage, register "down" stage
        if angle < upper_limit and stage == 'upper':
            stage = 'mid_down'
        if angle < angle_mid and stage == 'mid_down':
            if j == 0:
                image_middown = image
                j += 1

        # Below the lower limit and after the "down" stage, register the "hold down" stage
        if angle < lower_limit and stage =='mid_down':
            stage = 'lower'
            if k == 0:
                if angle < lower_limit:
                    image_lower = image
                    k += 1

        # Above the lower limit and after the "hold down" stage, register the "up" stage and count the rep
        if angle > lower_limit and stage == 'lower':
            stage = 'mid_up'
            if l == 0:
                if angle > lower_limit:
                    image_midup = image
                    l += 1
    else:
        print('This exercise type is not available')

    return stage, image_upper, image_middown, image_lower, image_midup, i,j,k,l










