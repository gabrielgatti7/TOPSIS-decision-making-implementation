# TOPSIS-decision-making-implementation
TOPSIS is a method for solving multi-criteria decision making problems.  
This is an implementation of the standart algorithm, which takes as input:
  - A matrix D whose rows are the alternatives and columns are the criteria (must be in a csv file)
  - A list of weights, indicating the weight of each criterion
  - A cost/benefit list, where each position indicates whether the corresponding criterion is cost or benefit

    
And generate as output:
  - A ranking graph of the best alternatives based on the closeness coefficient.
