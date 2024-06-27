import matplotlib.pyplot as plt

# Correcting the data formatting for plotting
sgd_data = {
    "Epochs": [1, 2, 3, 4, 5],
    "SimpleSGD": [2.576453421, 2.437521316, 2.342523772, 2.279932193, 2.23174],
    "SimpleSGDCurvature": [2.036710917, 1.650997667, 1.456857583, 1.310749333, 1.174719667]
}
heavyball_data = {
    "Epochs": [1, 2, 3, 4, 5],
    "HeavyBall": [2.256429083, 1.945472083, 1.793087833, 1.710580083, 1.63661875],
    "HeavyBallCurvature": [2.088772167, 1.708338167, 1.50364125, 1.332215667, 1.17257825]
}
nag_data = {
    "Epochs": [1, 2, 3, 4, 5],
    "NAG": [2.255914417, 1.945064667, 1.794625333, 1.714205417, 1.64251275],
    "NAGCurvature": [2.1006655, 1.737555417, 1.547732833, 1.394217833, 1.249776083]
}

colors = {
    "blue": "#4285F4",
    "red": "#EA4335",
    "yellow": "#FBBC05",
    "green": "#34A853"
}

# Creating subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plotting SGD data
axs[0].plot(sgd_data["Epochs"], sgd_data["SimpleSGD"], label='SimpleSGD', marker='o', color=colors["blue"])
axs[0].plot(sgd_data["Epochs"], sgd_data["SimpleSGDCurvature"], label='SimpleSGDCurvature', marker='s', color=colors["red"])
axs[0].set_title('SGD', fontsize=27)
axs[0].set_xlabel('Epoch', fontsize=17)
axs[0].set_ylabel('Mean Training Loss', fontsize=17)
axs[0].set_xticks(sgd_data["Epochs"])
axs[0].legend(fontsize=15)
axs[0].grid(True, which='both', axis='both')
axs[0].minorticks_off()

# Plotting HeavyBall data
axs[1].plot(heavyball_data["Epochs"], heavyball_data["HeavyBall"], label='HeavyBall', marker='o', color=colors["blue"])
axs[1].plot(heavyball_data["Epochs"], heavyball_data["HeavyBallCurvature"], label='HeavyBallCurvature', marker='s', color=colors["red"])
axs[1].set_title('HeavyBall', fontsize=27)
axs[1].set_xlabel('Epoch', fontsize=17)
axs[1].set_ylabel('Mean Training Loss', fontsize=17)
axs[1].set_xticks(heavyball_data["Epochs"])
axs[1].legend(fontsize=15)
axs[1].grid(True, which='both', axis='both')
axs[1].minorticks_off()

# Plotting NAG data
axs[2].plot(nag_data["Epochs"], nag_data["NAG"], label='NAG', marker='o', color=colors["blue"])
axs[2].plot(nag_data["Epochs"], nag_data["NAGCurvature"], label='NAGCurvature', marker='s', color=colors["red"])
axs[2].set_title('NAG', fontsize=27)
axs[2].set_xlabel('Epoch', fontsize=17)
axs[2].set_ylabel('Mean Training Loss', fontsize=17)
axs[2].set_xticks(nag_data["Epochs"])
axs[2].legend(fontsize=15)
axs[2].grid(True, which='both', axis='both')
axs[2].minorticks_off()



plt.tight_layout()
plt.savefig('outputs/plots/mean_training_loss_across_epochs.pdf')
plt.show()