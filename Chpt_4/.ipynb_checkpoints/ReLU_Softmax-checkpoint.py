import numpy as np

import os
import sys
module = os.path.abspath(os.path.join('..'))
if module not in sys.path:
    sys.path.append(module)

from Chpt_3.DenseLayers import Layer_Dense



class Activation_ReLU:
    
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)