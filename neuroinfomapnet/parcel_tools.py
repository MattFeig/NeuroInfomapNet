import pandas as pd
import numpy as np

# Reorder parcels by group average gordon network label
reorder_indices = [9,63,64,65,66,67,68,69,76,101,103,159,170,223,226,229,
                231,232,238,243,267,268,328,329,20,21,26,27,33,39,62,
                70,71,75,80,81,83,100,102,104,110,111,146,152,179,180,
                184,186,187,191,195,197,218,222,233,234,237,244,245,
                247,248,273,316,317,11,88,92,172,253,0,3,5,24,25,43,93,
                113,115,116,125,126,144,145,149,150,151,153,155,156,161,
                164,183,185,199,219,224,256,258,277,278,289,314,315,320,
                321,322,323,324,325,330,40,41,42,48,50,51,54,73,86,87,90,
                91,94,99,105,106,109,112,154,188,198,202,207,210,235,249,
                251,252,261,265,270,274,6,8,23,77,95,107,108,147,148,166,
                167,169,181,239,259,260,271,272,275,276,318,319,326,327,
                10,17,18,72,114,117,118,119,120,121,122,123,124,127,128,
                132,133,134,141,143,158,171,177,178,279,280,281,282,283,
                284,285,286,287,288,290,291,295,296,299,300,301,302,303,
                304,305,311,313,12,13,129,142,173,293,294,312,28,82,182,
                246,1,29,30,31,32,34,35,36,37,44,45,46,47,49,53,55,56,57,
                162,189,190,192,193,194,200,201,203,204,205,206,208,209,
                212,213,214,215,216,269,2,38,52,58,163,196,211,217,22,59,
                60,61,74,78,79,84,85,157,160,220,221,225,227,228,230,236,
                240,241,242,331,332,4,7,14,15,16,19,89,96,97,98,130,131,
                135,136,137,138,139,140,165,168,174,175,176,250,254,255,
                257,262,263,264,266,292,297,298,306,307,308,309,310]

parcel_labels = pd.read_excel('../data/parcel_information/Parcels.xlsx')

clist = ['red' if x == 'Default' else 'cyan' if x == 'SMhand' else 'orange' if x =='SMmouth' else 'blue' if x == 'Visual' else 'magenta'if x == 'Auditory' else 'purple' if x == "CinguloOperc" else 'yellow' if x =='FrontoParietal' else 'lime' if x == 'DorsalAttn' else 'teal' if x =='VentralAttn' else 'black' if x == 'Salience' else 'white' for x in parcel_labels['Community']]

none_inds = np.where(parcel_labels['Community'] == 'None')[0]

# Vertex wise networks ordered 0-17
net_list = np.array(["UNASSIGNED", "DMN", "VIS", "FPN", "MEDIALVIS", "DAN", "PREMOTOR", "VAN", "SAL", "CON", "SM-Body",
                      "SM-Face", "AUD", "MTL", "VTL", "MEM", "REW", "SM-Foot"])

color_list = np.array(['white', 'red', 'darkblue', 'yellow', 'grey', 'lime', 'grey', 'teal', 'black', 'purple', 'cyan',
                       'orange', 'darkviolet', 'grey', 'grey', 'grey', 'grey', 'darkgreen'])

# Vertex wise network indices subsetted for all existing ones, not inlcuding MEM or REW
subset_nets = [1,2,3,5,7,8,9,10,11,12,17]
man_order = [2,12,11,10,17,1,9,3,8,5,7]
sensory_inds = np.array([2,12,11,10,17])
con_att_inds = np.array([9,3,8,5,7]) # NO DMN

#  network  ordering
names_abbrev = ['Auditory','CingOperc','Memory','Default','DorsalAtt','FrontoPar','None',
        'Context','Salience','SM-Body','SM-Face','VentralAtt','Visual']

range_list = [(0,24),(24,64), (64,69), (69,110),(110,142),(142,166),(166,213),
        (213, 221),(221, 225),(225, 263),(263, 271),(271, 294),(294, 333)]

