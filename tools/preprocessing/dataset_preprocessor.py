'''
# Following CiA, we form chiral groups as (verb1, verb2, [shared_target]).
# Example:

# (“pick_needle”, “drop_needle”, [needle])

# (“push_needle”, “pull_suture”, [tissue_plane])

# (“grasp”, “release”, [needle])

# (“grasp”, “retract”, [gallbladder])

# (“coagulate”, “irrigate”, [liver])

# (“dissect”, “approximate”, [cystic_plate])


'''
Surgical_chirality = {
  "action_pairs": {
    # --- Strong temporal opposites ---
    "grasp": {
      "temporal_opposite": "release",
      "temporal_chirality": ["forward", "backward"],
      "spatial_chirality": ["left", "right", "unknown"],
      "weight": 1.0
    },
    "pick_needle": {
      "temporal_opposite": "drop_needle",
      "temporal_chirality": ["forward", "backward"],
      "spatial_chirality": ["left", "right", "unknown"],
      "weight": 1.0
    },
    "push_needle": {
      "temporal_opposite": "pull_suture",
      "temporal_chirality": ["forward", "backward"],
      "spatial_chirality": ["left", "right", "unknown"],
      "weight": 1.0
    },
    "retract": {
      "temporal_opposite": "release",
      "temporal_chirality": ["forward", "backward"],
      "spatial_chirality": ["left", "right", "unknown"],
      "weight": 1.0
    },
    "pack": {
      "temporal_opposite": "unpack",
      "temporal_chirality": ["forward", "backward"],
      "spatial_chirality": ["left", "right", "unknown"],
      "weight": 1.0
    },
     "pull_suture": {
      "temporal_opposite": "loosen_suture",
      "temporal_chirality": ["forward", "backward"],
      "spatial_chirality": ["left", "right", "both", "unknown"],
      "weight": 0.3
    },
    "cut": {
      "temporal_opposite": "continue_suture",  # semantically opposite
      "temporal_chirality": ["forward", "backward"],
      "spatial_chirality": ["left", "right", "unknown"],
      "weight": 0.3
    },
    "coagulate": {
      "temporal_opposite": "irrigate",  # not strict inverse but complementary
      "temporal_chirality": ["forward", "backward"],
      "spatial_chirality": ["left", "right", "unknown"],
      "weight": 0.3
    }
  }
}
    # --- Weaker spatial opposites ---}
Surgical_chirality = {
  "verb_chirality": {
    # --- Core temporal chirality groups ---
    "grasp": {
      "temporal_opposite": "release",
      "temporal_chirality": ["forward", "backward"]
    },
    "pick": {
      "temporal_opposite": "drop",
      "temporal_chirality": ["forward", "backward"]
    },
    "push": {
      "temporal_opposite": "pull",
      "temporal_chirality": ["forward", "backward"]
    },
    "tighten": {
      "temporal_opposite": "loosen",
      "temporal_chirality": ["tighten", "loosen"]
    },
    "tie": {
      "temporal_opposite": "untie",
      "temporal_chirality": ["forward", "backward"]
    },
    # --- Contextual chirality (complementary, not strict inverses) ---
    "cut": {
      "temporal_opposite": "continue_suture",
      "temporal_chirality": ["terminate", "continue"]
    },
    "coagulate": {
      "temporal_opposite": "irrigate",
      "temporal_chirality": ["apply_energy", "relieve_energy"]
    }
  },

  "spatial_chirality": ["left", "right", "both", "unknown"]
}


Surgical_chirality = {
  "action_pairs": {
    "grasp": {"opposite": "release", "spatial": ["left", "right","unknown"], "weight": 1.0 },
    "pick_needle": {"opposite": "drop_needle", "spatial": ["left", "right","unknown"], "weight": 1.0},
    "push_needle": {"opposite": "pull_suture", "spatial": ["left", "right","unknown"], "weight": 1.0},
  
    "retract": {"opposite": "release", "spatial": ["left", "right","unknown"], "weight": 1.0},
    "tie_knot": {"opposite": "untie_knot", "spatial": ["both","unknown"], "weight": 1.0},
    "pack": {"opposite": "unpack", "spatial": ["left", "right","unknown"], "weight": 1.0},

    "reach": {"opposite": "return", "spatial": ["left", "right","unknown"], "weight": 0.5},
    "dissect": {"opposite": "approximate", "spatial": ["left", "right","unknown"], "weight": 0.5},
    "transfer_needle": {"opposite": "transfer_back", "spatial": ["L_to_R", "R_to_L","unknown"], "weight": 0.5},
    "clip": {"opposite": "unclip", "spatial": ["left", "right","unknown"], "weight": 0.5},
    "pull_suture": {"opposite": "loosen_suture", "spatial": ["left", "right", "unknown"], "weight": 0.0},
    "cut": {"opposite": "suture_continue", "spatial": ["left", "right","unknown"], "weight": 0.0},
    "coagulate": {"opposite": "irrigate", "spatial": ["left", "right","unknown"], "weight": 0.0},
    
  }
}

JIGSAWS_GESTURE_MAP = {
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
JIGSAWS_TO_IVT_CHIRALITY = {
    "G1":  ("needle_driver", "reach_needle", "needle", "right"),
    "G2":  ("needle_driver", "position_needle", "needle", "right"),
    "G3":  ("needle_driver", "push_needle", "tissue_plane", "right"),
    "G4":  ("needle_driver", "transfer_needle", "needle", "L_to_R"),
    "G5":  ("needle_driver", "move_center", "needle", "right"),
    "G6":  ("grasper", "pull_suture", "suture", "left"),
    "G7":  ("needle_driver", "pull_suture", "suture", "right"),
    "G8":  ("needle_driver", "orient_needle", "needle", "right"),
    "G9":  ("needle_driver", "tie_knot", "suture", "both"),
    "G10": ("grasper", "loosen_suture", "suture", "left"),
    "G11": ("grasper", "drop_needle", "needle", "left"),  # or suture? context-dependent
    "G12": ("grasper", "reach_needle", "needle", "left"),
    "G13": ("needle_driver", "tie_knot", "suture", "both"),
    "G14": ("needle_driver", "reach_needle", "suture", "right"),
    "G15": ("grasper", "pull_suture", "suture", "both")
}
# (Note: Skill/GRS data is metadata)

# RARP
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

RARP_TO_IVT_CHIRALITY = {
    0: ("grasper", "null_verb", "null_target", "unknown"),
    1: ("needle_driver", "pick_needle", "needle", "unknown"),
    2: ("needle_driver", "position_needle", "needle", "unknown"),
    3: ("needle_driver", "push_needle", "tissue_plane", "unknown"),
    4: ("needle_driver", "pull_suture", "tissue_plane", "unknown"),
    5: ("needle_driver", "tie_knot", "suture", "unknown"),  # though "both" is likely
    6: ("scissors", "cut_suture", "suture", "unknown"),
    7: ("needle_driver", "drop_needle", "needle", "unknown")
}

"reaching","positioning","pushing", "transferring", "moving_center","pulling",

chirality_axes = {
    "temporal": ["forward", "backward", "terminate", "continue", "tighten", "loosen", "unknown"],
    "spatial": ["left", "right", "both", "L_to_R", "R_to_L", "unknown"]
}
verb_chirality = {
    "grasp": "release",
    "pick": "drop",
    "push": "pull",
    "tie": "untie",
    "tighten": "loosen",
    "pack": "unpack",
    "clip": "unclip",

    # contextual opposites
    "cut": "continue_suture",
    "coagulate": "irrigate",
    "transfer": "transfer_back"
}


Actions_categories= {
        "instrument": {
            "0": "grasper",
            "1": "bipolar",
            "2": "hook",
            "3": "scissors",
            "4": "clipper",
            "5": "irrigator",
            "6": "needle_driver"
        },
        "verb": {
            "0": "grasp",
            "1": "retract",
            "2": "dissect",
            "3": "coagulate",
            "4": "clip",
            "5": "cut",
            "6": "aspirate",
            "7": "irrigate",
            "8": "pack",
            "9": "null_verb",
            "10": "pick",
            "11": "reach",
            "12": "push",
            "13": "pull",
            "14": "transfer",
            "15": "tie_knot",
            "16": "cut_suture",
            "17": "drop",
            "18": "position",
            "19": "orient",
            "20": "loosen_suture",
            "21": "move_center",
            "22": "approximate",
            "23": "release",
            "24": "unclip",
            "25": "untie_knot",
            "26": "continue_suture",
            "27": "transfer_back"
        },
        "target": {
            "0": "gallbladder",
            "1": "cystic_plate",
            "2": "cystic_duct",
            "3": "cystic_artery",
            "4": "cystic_pedicle",
            "5": "blood_vessel",
            "6": "fluid",
            "7": "abdominal_wall_cavity",
            "8": "liver",
            "9": "adhesion",
            "10": "omentum",
            "11": "peritoneum",
            "12": "gut",
            "13": "specimen_bag",
            "14": "null_target",
            "15": "needle",
            "16": "suture",
            "17": "tissue_plane",
            "18": "knot"
        },
        "triplet": {
            "0": "grasper,dissect,cystic_plate",
            "1": "grasper,dissect,gallbladder",
            "2": "grasper,dissect,omentum",
            "3": "grasper,grasp,cystic_artery",
            "4": "grasper,grasp,cystic_duct",
            "5": "grasper,grasp,cystic_pedicle",
            "6": "grasper,grasp,cystic_plate",
            "7": "grasper,grasp,gallbladder",
            "8": "grasper,grasp,gut",
            "9": "grasper,grasp,liver",
            "10": "grasper,grasp,omentum",
            "11": "grasper,grasp,peritoneum",
            "12": "grasper,grasp,specimen_bag",
            "13": "grasper,pack,gallbladder",
            "14": "grasper,retract,cystic_duct",
            "15": "grasper,retract,cystic_pedicle",
            "16": "grasper,retract,cystic_plate",
            "17": "grasper,retract,gallbladder",
            "18": "grasper,retract,gut",
            "19": "grasper,retract,liver",
            "20": "grasper,retract,omentum",
            "21": "grasper,retract,peritoneum",
            "22": "bipolar,coagulate,abdominal_wall_cavity",
            "23": "bipolar,coagulate,blood_vessel",
            "24": "bipolar,coagulate,cystic_artery",
            "25": "bipolar,coagulate,cystic_duct",
            "26": "bipolar,coagulate,cystic_pedicle",
            "27": "bipolar,coagulate,cystic_plate",
            "28": "bipolar,coagulate,gallbladder",
            "29": "bipolar,coagulate,liver",
            "30": "bipolar,coagulate,omentum",
            "31": "bipolar,coagulate,peritoneum",
            "32": "bipolar,dissect,adhesion",
            "33": "bipolar,dissect,cystic_artery",
            "34": "bipolar,dissect,cystic_duct",
            "35": "bipolar,dissect,cystic_plate",
            "36": "bipolar,dissect,gallbladder",
            "37": "bipolar,dissect,omentum",
            "38": "bipolar,grasp,cystic_plate",
            "39": "bipolar,grasp,liver",
            "40": "bipolar,grasp,specimen_bag",
            "41": "bipolar,retract,cystic_duct",
            "42": "bipolar,retract,cystic_pedicle",
            "43": "bipolar,retract,gallbladder",
            "44": "bipolar,retract,liver",
            "45": "bipolar,retract,omentum",
            "46": "hook,coagulate,blood_vessel",
            "47": "hook,coagulate,cystic_artery",
            "48": "hook,coagulate,cystic_duct",
            "49": "hook,coagulate,cystic_pedicle",
            "50": "hook,coagulate,cystic_plate",
            "51": "hook,coagulate,gallbladder",
            "52": "hook,coagulate,liver",
            "53": "hook,coagulate,omentum",
            "54": "hook,cut,blood_vessel",
            "55": "hook,cut,peritoneum",
            "56": "hook,dissect,blood_vessel",
            "57": "hook,dissect,cystic_artery",
            "58": "hook,dissect,cystic_duct",
            "59": "hook,dissect,cystic_plate",
            "60": "hook,dissect,gallbladder",
            "61": "hook,dissect,omentum",
            "62": "hook,dissect,peritoneum",
            "63": "hook,retract,gallbladder",
            "64": "hook,retract,liver",
            "65": "scissors,coagulate,omentum",
            "66": "scissors,cut,adhesion",
            "67": "scissors,cut,blood_vessel",
            "68": "scissors,cut,cystic_artery",
            "69": "scissors,cut,cystic_duct",
            "70": "scissors,cut,cystic_plate",
            "71": "scissors,cut,liver",
            "72": "scissors,cut,omentum",
            "73": "scissors,cut,peritoneum",
            "74": "scissors,dissect,cystic_plate",
            "75": "scissors,dissect,gallbladder",
            "76": "scissors,dissect,omentum",
            "77": "clipper,clip,blood_vessel",
            "78": "clipper,clip,cystic_artery",
            "79": "clipper,clip,cystic_duct",
            "80": "clipper,clip,cystic_pedicle",
            "81": "clipper,clip,cystic_plate",
            "82": "irrigator,aspirate,fluid",
            "83": "irrigator,dissect,cystic_duct",
            "84": "irrigator,dissect,cystic_pedicle",
            "85": "irrigator,dissect,cystic_plate",
            "86": "irrigator,dissect,gallbladder",
            "87": "irrigator,dissect,omentum",
            "88": "irrigator,irrigate,abdominal_wall_cavity",
            "89": "irrigator,irrigate,cystic_pedicle",
            "90": "irrigator,irrigate,liver",
            "91": "irrigator,retract,gallbladder",
            "92": "irrigator,retract,liver",
            "93": "irrigator,retract,omentum",
            "94": "grasper,null_verb,null_target",
            "95": "bipolar,null_verb,null_target",
            "96": "hook,null_verb,null_target",
            "97": "scissors,null_verb,null_target",
            "98": "clipper,null_verb,null_target",
            "99": "irrigator,null_verb,null_target",
            "100": "needle_driver,pick_needle,needle",
            "101": "needle_driver,reach_needle,needle",
            "102": "needle_driver,push_needle,tissue_plane",
            "103": "needle_driver,pull_suture,suture",
            "104": "needle_driver,transfer_needle,needle",
            "105": "needle_driver,tie_knot,suture",
            "106": "scissors,cut_suture,suture",
            "107": "hook,cut_suture,suture",
            "108": "grasper,pull_suture,suture",
            "109": "needle_driver,orient_needle,needle",
            "110": "grasper,loosen_suture,suture",
            "111": "needle_driver,position_needle,needle",
            "112": "needle_driver,drop_needle,needle",
            "113": "grasper,reach_needle,needle",
            "114": "grasper,move_center,needle",
            "115": "needle_driver,reach_needle,suture",        # JIGSAWS G14
            "116": "needle_driver,pull_suture,tissue_plane",   # RARP4
            "117": "needle_driver,approximate,tissue_plane",
            "118": "grasper,release,needle",
        },
        "phase": {
            "0": "preparation",
            "1": "carlot-triangle-dissection",
            "2": "clipping-and-cutting",
            "3": "gallbladder-dissection",
            "4": "gallbladder-packaging",
            "5": "cleaning-and-coagulation",
            "6": "gallbladder-extraction",
            "7": "suturing"
        }
    },

# Reverse lookup for IDs
instrument_to_id = {v: k for k, v in Actions_categories["instrument"].items()}
verb_to_id = {v: k for k, v in Actions_categories["verb"].items()}
target_to_id = {v: k for k, v in Actions_categories["target"].items()}

# Triplet string → triplet_id reverse map
triplet_str_to_id = {v: k for k, v in Actions_categories["triplet"].items()}
