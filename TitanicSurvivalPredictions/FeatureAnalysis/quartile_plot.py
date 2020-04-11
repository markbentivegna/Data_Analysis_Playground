import matplotlib.pyplot as plt

def generate_quartile_plot(quartile_data):
    ax1 = plt.subplots()
    ax1.boxplot(quartile_data)
    plt.show()