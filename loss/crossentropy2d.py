#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "3DPerceptionLAB"
__copyright__ = "Copyright 2018, 3DPerceptionLAB"
__license__ = "GPL-3.0"
__version__ = "1.0.1"
__maintainer__ = "Sergiu-Ovidiu Oprea"
__email__ = "soprea@dtic.ua.es"
__status__ = "Production"


import torch

class CrossEntropy2D(torch.nn.Module):

	def __init__(self, weight= None, size_average= True, ignore_index= 255):
		super().__init__()

		self.weight= weight
		self.size_average= size_average
		print(ignore_index)
		self.ignore_index= ignore_index
		self.loss= torch.nn.CrossEntropyLoss(self.weight, self.size_average, self.ignore_index)

	def __repr__(self):

		return "Cross-entropy 2D loss with the following parameteres: \n -> weight {0} \n -> size_average {1} \n \
		-> ignore_index {2} \n".format(self.weight, self.size_average, self.ignore_index)

	def forward(self, inputs, targets):
		return self.loss(inputs, targets)


