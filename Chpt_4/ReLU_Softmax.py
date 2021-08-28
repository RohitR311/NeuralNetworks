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
        
class Activation_Softmax:
    
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilties = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilties