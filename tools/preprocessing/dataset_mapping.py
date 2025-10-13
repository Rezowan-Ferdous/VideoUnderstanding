JIGSAWS_GESTURE_MAP = { "G1": "reaching for the needle with right hand", "G2": "positioning the tip of the needle",
                       "G3": "pushing needle through the tissue", "G4": "transferring needle from left to right",
                        "G5": "moving to center of workspace with needle in grip", "G6": "pulling suture with left hand",
                        "G7": "pulling suture with right hand", "G8": "orienting needle", "G9": "using right hand to help tighten suture",
                        "G10": "loosening more suture", "G11": "dropping suture and moving to end points", "G12": "reaching for needle with left hand",
                        "G13": "making C loop around right hand", "G14": "reaching for suture with right hand", "G15": "pulling suture with both hands" }
RARP_ACTION_ID_MAP = { 0: "Preparation", 1: "Picking up the needle", 2: "Positioning the needle tip", 3: "Pushing the needle through the tissue",
                      4: "Pulling the needle out of the tissue", 5: "Tying a knot", 6: "Cutting the suture", 7: "Returning/dropping the needle", }



null_verb_target= [-1,94,95,96,97,98,99]
chirality_verb_pairs = {
    "0": "10",  # grasp <-> release (release is hypothetical ID 10)
    "11":"12", # pick drop
    "16":"17", # push pull
    "18": "19", # tie <-> untie (from your example, hypothetical IDs)
    "21": "20", # "tighten": "loosen",
    # "4": "20",  # clip <-> unclip (unclip is hypothetical ID 20)
    # "5": "21",  # cut <-> suture (suture is hypothetical ID 21)
}
instrument_dict = {
    0: "grasper",
    1: "bipolar",
    2: "hook",
    3: "scissors",
    4: "clipper",
    5: "irrigator",
    6: "needle_driver"
}

verb_dict = {
0: "grasp",
1: "retract",
2: "dissect",
3: "coagulate",
4: "clip",
5: "cut",
6: "aspirate",
7: "irrigate",
8: "pack",
9: "null_verb",
10: "release",
11: "pick",
12: "drop",
13: "reach",
14: "transfer",
15: "transfer_back",
16: "push",
17: "pull",
18: "tie",
19: "untie",
20: "loosen",
21: "tighten",
22: "orient",
23: "position",
24: "approximate",
25: "continue_suture",
26: "move_center"
},

target_dict= {
0: "gallbladder",
1: "cystic_plate",
2: "cystic_duct",
3: "cystic_artery",
4: "cystic_pedicle",
5: "blood_vessel",
6: "fluid",
7: "abdominal_wall_cavity",
8: "liver",
9: "adhesion",
10: "omentum",
11: "peritoneum",
12: "gut",
13: "specimen_bag",
14: "null_target",
15: "needle",
16: "suture",
17: "tissue_plane",
18: "knot"
},

triplet_dict= {
0: "grasper,dissect,cystic_plate",
1: "grasper,dissect,gallbladder",
2: "grasper,dissect,omentum",
3: "grasper,grasp,cystic_artery",
4: "grasper,grasp,cystic_duct",
5: "grasper,grasp,cystic_pedicle",
6: "grasper,grasp,cystic_plate",
7: "grasper,grasp,gallbladder",
8: "grasper,grasp,gut",
9: "grasper,grasp,liver",
10: "grasper,grasp,omentum",
11: "grasper,grasp,peritoneum",
12: "grasper,grasp,specimen_bag",
13: "grasper,pack,gallbladder",
14: "grasper,retract,cystic_duct",
15: "grasper,retract,cystic_pedicle",
16: "grasper,retract,cystic_plate",
17: "grasper,retract,gallbladder",
18: "grasper,retract,gut",
19: "grasper,retract,liver",
20: "grasper,retract,omentum",
21: "grasper,retract,peritoneum",
22: "bipolar,coagulate,abdominal_wall_cavity",
23: "bipolar,coagulate,blood_vessel",
24: "bipolar,coagulate,cystic_artery",
25: "bipolar,coagulate,cystic_duct",
26: "bipolar,coagulate,cystic_pedicle",
27: "bipolar,coagulate,cystic_plate",
28: "bipolar,coagulate,gallbladder",
29: "bipolar,coagulate,liver",
30: "bipolar,coagulate,omentum",
31: "bipolar,coagulate,peritoneum",
32: "bipolar,dissect,adhesion",
33: "bipolar,dissect,cystic_artery",
34: "bipolar,dissect,cystic_duct",
35: "bipolar,dissect,cystic_plate",
36: "bipolar,dissect,gallbladder",
37: "bipolar,dissect,omentum",
38: "bipolar,grasp,cystic_plate",
39: "bipolar,grasp,liver",
40: "bipolar,grasp,specimen_bag",
41: "bipolar,retract,cystic_duct",
42: "bipolar,retract,cystic_pedicle",
43: "bipolar,retract,gallbladder",
44: "bipolar,retract,liver",
45: "bipolar,retract,omentum",
46: "hook,coagulate,blood_vessel",
47: "hook,coagulate,cystic_artery",
48: "hook,coagulate,cystic_duct",
49: "hook,coagulate,cystic_pedicle",
50: "hook,coagulate,cystic_plate",
51: "hook,coagulate,gallbladder",
52: "hook,coagulate,liver",
53: "hook,coagulate,omentum",
54: "hook,cut,blood_vessel",
55: "hook,cut,peritoneum",
56: "hook,dissect,blood_vessel",
57: "hook,dissect,cystic_artery",
58: "hook,dissect,cystic_duct",
59: "hook,dissect,cystic_plate",
60: "hook,dissect,gallbladder",
61: "hook,dissect,omentum",
62: "hook,dissect,peritoneum",
63: "hook,retract,gallbladder",
64: "hook,retract,liver",
65: "scissors,coagulate,omentum",
66: "scissors,cut,adhesion",
67: "scissors,cut,blood_vessel",
68: "scissors,cut,cystic_artery",
69: "scissors,cut,cystic_duct",
70: "scissors,cut,cystic_plate",
71: "scissors,cut,liver",
72: "scissors,cut,omentum",
73: "scissors,cut,peritoneum",
74: "scissors,dissect,cystic_plate",
75: "scissors,dissect,gallbladder",
76: "scissors,dissect,omentum",
77: "clipper,clip,blood_vessel",
78: "clipper,clip,cystic_artery",
79: "clipper,clip,cystic_duct",
80: "clipper,clip,cystic_pedicle",
81: "clipper,clip,cystic_plate",
82: "irrigator,aspirate,fluid",
83: "irrigator,dissect,cystic_duct",
84: "irrigator,dissect,cystic_pedicle",
85: "irrigator,dissect,cystic_plate",
86: "irrigator,dissect,gallbladder",
87: "irrigator,dissect,omentum",
88: "irrigator,irrigate,abdominal_wall_cavity",
89: "irrigator,irrigate,cystic_pedicle",
90: "irrigator,irrigate,liver",
91: "irrigator,retract,gallbladder",
92: "irrigator,retract,liver",
93: "irrigator,retract,omentum",
94: "grasper,null_verb,null_target",
95: "bipolar,null_verb,null_target",
96: "hook,null_verb,null_target",
97: "scissors,null_verb,null_target",
98: "clipper,null_verb,null_target",
99: "irrigator,null_verb,null_target",
100: "needle_driver,pick,needle",   # RARP1
101: "needle_driver,position,needle",  # RARP2
102: "needle_driver,push,tissue_plane", # RARP3
103: "needle_driver,transfer,needle",
104: "needle_driver,move_center,needle",
105: "grasper,pull,suture", # G15  
106: "needle_driver,pull,suture",
107: "needle_driver,orient,needle",
108: "needle_driver,tighten,suture",
109: "grasper,loosen,suture",
110: "grasper,drop,suture",
111: "grasper,pick,needle",
112: "needle_driver,tie,knot",  # RARP5
113: "needle_driver,pick,suture",  
114: "needle_driver,pull,tissue_plane",      # RARP4      
115: "scissors,cut,suture",                  # RARP6
116: "needle_driver,drop,needle",            # RARP7
117: "grasper,pick,suture",
}

phase_dict= {
0: "preparation",
1: "carlot-triangle-dissection",
2: "clipping-and-cutting",
3: "gallbladder-dissection",
4: "gallbladder-packaging",
5: "cleaning-and-coagulation",
6: "gallbladder-extraction"
}

gesture_map = {
    "G1": "reaching for the needle with right hand",
    "G2": "positioning the tip of the needle", 
    "G3": "pushing needle through the tissue",
    "G4": "transferring needle from left to right",
    "G5": "moving to center of workspace with needle in grip",
    "G6": "pulling suture with left hand",
    "G7": "pulling suture with right hand",
    "G8": "orienting needle",
    "G9": "using right hand to help tighten suture",
    "G10": "loosening more suture",
    "G11": "dropping suture and moving to end points",
    "G12": "reaching for needle with left hand",
    "G13": "making C loop around right hand",
    "G14": "reaching for suture with right hand",
    "G15": "pulling suture with both hands"
}

RARP_ACTION_ID_MAP = {
    0: "Other",
    1: "Picking up the needle",
    2: "Positioning the needle tip",
    3: "Pushing the needle through the tissue",
    4: "Pulling the needle out of the tissue",
    5: "Tying a knot",
    6: "Cutting the suture",
    7: "Returning/dropping the needle",
}

JIGSAWS_TO_TRIPLET = {
    "G1": [111],         # reaching needle RH → grasper,pick,needle
    "G2": [101],         # positioning tip → needle_driver,position,needle
    "G3": [102],         # pushing needle → needle_driver,push,tissue_plane
    "G4": [103],         # transfer needle → needle_driver,transfer,needle
    "G5": [104],         # move center → needle_driver,move_center,needle
    "G6": [105],         # pull suture LH → grasper,pull,suture
    "G7": [106],         # pull suture RH → needle_driver,pull,suture
    "G8": [107],         # orient needle
    "G9": [108],         # tighten suture RH → needle_driver,tighten,suture
    "G10": [109],        # loosen suture
    "G11": [110],        # drop suture
    "G12": [111],        # reaching needle LH
    "G13": [112],           # making C loop around hand tie knot
    "G14": [112],        # reaching for suture → needle_driver,pick,suture
    "G15": [112],   # pulling with both hands → grasper+needle_driver
}
RARP_TO_TRIPLET = {
    1: [100],  # pick up needle
    2: [101],  # position needle
    3: [102],  # push needle through tissue
    4: [114],  # pull needle out
    5: [112],  # tie knot
    6: [115],  # cut suture
    7: [116],  # return/drop needle
}
def get_triplets_from_gesture(gesture_id):
    return [triplet_dict[t] for t in JIGSAWS_TO_TRIPLET.get(gesture_id, [])]

def get_triplets_from_rarp(rarp_id):
    return [triplet_dict[t] for t in RARP_TO_TRIPLET.get(rarp_id, [])]

def get_gestures_from_triplet(triplet_id):
    return [g for g, ts in JIGSAWS_TO_TRIPLET.items() if triplet_id in ts]

def get_rarp_from_triplet(triplet_id):
    return [r for r, ts in RARP_TO_TRIPLET.items() if triplet_id in ts]