import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class TOPSIS:

    def __init__(self, d_matrix, cost_benefit, weights=None, alt_names=None, dist_name='euclidean', normalize=True):
        """
        Parameters
        ----------
        d_matrix : str
            The path to the csv file containing the decision matrix. 
            This file must have the alternatives as rows and the criteria as columns.
        
        cost_benefit : list
            A list containing the cost or benefit of each criteria. The values must be either 'c' or 'b'.
        
        weights : list, None, optional
            A list containing the weights of each criteria. The default is None. 
            If None, the weights will be equally distributed.
        
        alt_names : list, optional
            A list containing the names of the alternatives. The default is None.
            
        dist_name : str, optional
            The name of the distance metric to be used. The default is 'euclidean'.
        
        normalize : str, optional
            A string indicating if the decision matrix should be normalized. The default is 'True'.
        """

        if not isinstance(d_matrix, str):
            raise TypeError("d_matrix must be a csv file")
        
        # Check if the first line of the csv file is a header
        has_header = False
        with open(d_matrix, 'r') as file:
            first_line = file.readline()
            try:
                # Try to convert the first line to float
                float_vals = [float(val) for val in first_line.split(',')]
                has_header = False
            except ValueError:
                has_header = True

        if has_header:
            self.d_matrix = np.genfromtxt(d_matrix, delimiter=',', skip_header=1)
        else:
            self.d_matrix = np.genfromtxt(d_matrix, delimiter=',')

        # Getting the number of alternative and criteria
        self.n_alt, self.n_crit = self.d_matrix.shape

        if weights is None:
            self.weights = np.array([1] * self.n_crit) / self.n_crit
        else:
            if not isinstance(weights, list) and not isinstance(weights, np.ndarray):
                raise ValueError(f"The weights must be either a list or a Numpy array.")
            elif len(weights) != self.n_crit:
                raise ValueError("The number of weights must be the same as the number of criteria")

            self.weights = np.asarray(weights)
            # Check if the weights are normalized, if not normalize them
            if not np.isclose(self.weights.sum(), 1.0):
                self.weights = self.weights / self.weights.sum()
                print("The weights were normalized within the interval [0,1]")
        
        if alt_names is None:
            self.alt_names = [f"A{i}" for i in range(self.n_alt)]
        else:
            if not isinstance(alt_names, list):
                raise ValueError("The alt_names must be a list")
            if len(alt_names) != self.n_alt:
                raise ValueError("The number of alt_names must be the same as the number of alternatives")
            self.alt_names = alt_names

        self.dist_name = dist_name
        self.normalize = normalize

        if not isinstance(cost_benefit, list):
            raise ValueError("The cost_benefit must be a list")
        if len(cost_benefit) != self.n_crit:
            raise ValueError("The number of cost_benefit must be the same as the number of criteria")
        self.cost_benefit = cost_benefit
        
        # Normalize the decision matrix
        if self.normalize:
            self.normalize_d_matrix()
        
        # Apply wheigts to the decision matrix
        self.d_matrix = self.d_matrix * self.weights

        self.ideal_sol = np.zeros(self.n_crit, dtype=float)
        self.anti_ideal_sol = np.zeros(self.n_crit, dtype=float)
        self.dist_ideal = np.zeros(self.n_alt, dtype=float)
        self.dist_anti_ideal = np.zeros(self.n_alt, dtype=float)
        self.closeness_coef = np.zeros(self.n_alt, dtype=float)

    def normalize_d_matrix(self):
        """
        Normalize the decision matrix following the standard TOPSIS algorithm.
        """
        m = self.d_matrix ** 2
        m = np.sqrt(m.sum(axis=0))
        self.d_matrix = self.d_matrix / m

    def calculate_distances_to_ideal(self):
        """
        Calculate the distances to the ideal and anti-ideal solutions. 
        Results are stored in the attributes dist_ideal and dist_anti_ideal.
        """
        for i in range(self.n_alt):
            for j in range(self.n_crit):
                self.dist_ideal[i] += distance(self.d_matrix[i, j], self.ideal_sol[j], self.dist_name)
                self.dist_anti_ideal[i] += distance(self.d_matrix[i, j], self.anti_ideal_sol[j], self.dist_name)
            
            self.dist_ideal[i] = np.sqrt(self.dist_ideal[i])
            self.dist_anti_ideal[i] = np.sqrt(self.dist_anti_ideal[i])

    def generate_closeness_coef(self):
        """
        Generate the closeness coefficient for each alternative. 
        Result is stored in the closeness_coef attribute.
        """
        # Calculate the ideal and anti-ideal solutions
        for i in range(self.n_crit):
            if self.cost_benefit[i] == 'c':
                self.ideal_sol[i] = self.d_matrix[:, i].min()
                self.anti_ideal_sol[i] = self.d_matrix[:, i].max()
            elif self.cost_benefit[i] == 'b':
                self.ideal_sol[i] = self.d_matrix[:, i].max()
                self.anti_ideal_sol[i] = self.d_matrix[:, i].min()
            else:
                raise ValueError("The cost_benefit must be either 'c' or 'b'")
            
        self.calculate_distances_to_ideal()
        
        # Calculate the closeness coefficient
        for i in range(self.n_alt):
            self.closeness_coef[i] = self.dist_anti_ideal[i] / (self.dist_ideal[i] + self.dist_anti_ideal[i])

    def print_inputs(self):
        """
        Print the inputs of the TOPSIS algorithm (decision matrix, weights and cost and benefit array).
        """
        print('-' * 20)
        print('Decision Matrix:')
        print(self.d_matrix)
        print('-' * 20)

        print('Weights:')
        print(self.weights)
        print('-' * 20)

        print('Cost and benefit:')
        print(self.cost_benefit)
        print('-' * 20)

    def plot_ranking(self):
        """
        Plot the ranking of the alternatives.
        """
        p = sns.barplot(x=self.alt_names, y=self.closeness_coef)
        p.set_xlabel("Alternatives")
        p.set_ylabel("Closeness Coefficient")
        plt.show()


# Static method
def distance(a, b, dist_name='euclidean'):
    """
    Calculate the distance between two values. Used by TOPSIS.calculate_distances()

    Parameters
    ----------
    a : float
        The first value to calculate the distance.

    b : float
        The second value to calculate the distance.

    dist_name : str, optional
        The name of the distance metric to be used. The default is 'euclidean'.
    """

    if dist_name == 'euclidean':
        return (a - b) ** 2
    else:
        raise ValueError("Only the euclidean distance is supported at the moment")