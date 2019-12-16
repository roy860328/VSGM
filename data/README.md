# Dataset

## Download

## File Structure

## JSON Structure

Each expert demonstration contains 3 or more language annotations in `traj_data.json`:

Task Info:
```
['task_id'] = "trial_00003_T20190312_234237"        (unique trajectory ID)
['task_type'] = "pick_heat_then_place_in_recep"     (one of 7 task types)
['pddl_params'] = {'object_target': "AlarmClock",   (object)
                   'parent_target': "DeskLamp",     (receptacle)
                   'mrecep_target': "",             (movable receptacle)
                   "toggle_target": "",             (toggle object)
                   "object_sliced": false}          (should the object be sliced?)
```

Scene Info:
```
['scene'] =  {'floor_plan': "FloorPlan7",           (THOR scene name)
              'scene_num' : 7,                      (THOR scene number)
              'random_seed': 3810970210,            (seed for initializing object placements)
              'init_action' : <API_CMD>,            (called to set the starting position of the agent)
              'object_poses': <LIST_OBJS>,          (initial 6DOF poses of objects in the scene)
              'object_toggles': <LIST_OBJS>}        (initial states of togglable objects)
```

Language Annotations:
```
['turk_annotations']['anns'] =  
             [{'task_desc': "Examine a clock using the light of a lamp.",                      (goal instruction) 
               'high_descs': ["Turn to the left and move forward to the window ledge." ...],   (list of step-by-step instructions)
               'votes': [1, 1, 1]                                                              (AMTurk languauge quality votes)
              },
              ...]
```

Expert Demonstration:
```
['plan'] = {'high_pddl':
                ...,
                ["high_idx": 4,                          (high-level subgoal index)
                 "discrete_action":                    
                     {"action": "PutObject",             (discrete high-level action)
                      "args": ["bread", "microwave"],    (discrete params)
                 "planner_action": <PDDL_ACTION> ],      (PDDL action)
                ...],
                 
            'low_actions': 
                ...,
                ["high_idx": 1,                          (high-level subgoal index)
                 "discrete_action":
                     {"action": "PickupObject",          (discrete low-level action)
                      "args": 
                          {"bbox": [180, 346, 332, 421]} (bounding box for interact action)
                           "mask": [0, 0, ... 1, 1]},    (compressed pixel mask for interact action)
                 "api_action": <API_CMD> ],              (THOR API command for replay)
                ...], 
           }
```

Images:
```
['images'] = [{"low_idx": 0,                    (low-level action index)
               "high_idx": 0,                   (high-level action index)
               "image_name": "000000000.png"}   (image filename)
             ...]
```