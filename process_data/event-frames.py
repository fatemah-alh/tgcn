#%%
from PIL import Image
import numpy as np
import pandas as pd
import yaml
import sys
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from utiles import get_file_names
from tqdm import tqdm 
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import imageio
import torch





#%%
np.mean([
    0.9, 1.0, 0.65, 0.87, 0.8717948794364929, 0.9729729890823364, 
    0.7749999761581421, 0.5263158082962036, 0.8461538553237915, 0.5, 
    0.824999988079071, 0.9750000238418579, 0.7250000238418579, 0.8717948794364929, 
    0.5, 0.7894737124443054, 0.5, 0.8717948794364929, 0.8999999761581421, 1.0, 
    0.6410256624221802, 0.7575757503509521, 0.6410256624221802, 0.9259259104728699, 
    0.8709677457809448, 0.5, 0.5405405163764954, 0.5, 0.5128205418586731, 
    0.9696969985961914, 1.0, 0.5151515007019043, 0.5945945978164673, 
    0.8717948794364929, 0.8000000715255737, 0.6410256624221802, 0.625, 0.5, 
    0.925000011920929, 0.5161290168762207, 0.6666666865348816, 0.7058823704719543, 
    0.7368420958518982, 0.8461538553237915, 1.0, 0.9090909361839294, 0.5, 
    0.625, 0.5333333611488342, 0.6000000238418579, 0.699999988079071, 
    0.9750000238418579, 0.625, 0.625, 0.625, 0.5, 0.5, 0.4871794879436493, 
    0.8500000238418579, 0.8055556416511536, 0.9750000238418579, 0.5, 
    0.574999988079071, 0.6666666865348816, 0.8461538553237915, 0.824999988079071, 
    0.5263158082962036
]
)
np.mean([
    0.9750000238418579, 0.6000000238418579, 0.949999988079071, 0.692307710647583, 
    0.9459459185600281, 0.7749999761581421, 0.7368420958518982, 0.692307710647583, 
    0.824999988079071, 0.5, 0.5, 0.5, 0.8974359035491943, 0.949999988079071, 
    0.5, 0.5, 0.8205128312110901, 0.925000011920929, 1.0, 0.692307710647583, 
    0.5151515007019043, 0.6666666865348816
]
)

#%%
np.mean([
    0.25, 0.47999998927116394, 0.28, 0.3100000023841858, 0.3232323229312897, 
    0.3645833432674408, 0.2142857164144516, 0.27551019191741943, 0.3263157904148102, 
    0.23000000417232513, 0.28999999165534973, 0.2800000011920929, 0.20000001788139343, 
    0.2083333283662796, 0.20000001788139343, 0.2857142984867096
]
)

# %%
np.mean([
    0.41999998688697815, 0.23999999463558197, 0.28999999165534973, 
    0.31313130259513855, 0.3020833432674408, 0.26530611515045166, 
    0.27551019191741943
]
)
# %%
