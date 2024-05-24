from topsis import TOPSIS

d_matrix = 'test/d_matrix.csv'
cost_benefit = ['c', 'b', 'c', 'b']
weights = [0.3, 0.05, 0.6, 0.05]
alt_names = ['Palio', 'HB20', 'Corolla']
tp = TOPSIS(d_matrix, cost_benefit, weights=weights, alt_names=alt_names)
tp.print_inputs()
tp.generate_closeness_coef()
tp.plot_ranking()