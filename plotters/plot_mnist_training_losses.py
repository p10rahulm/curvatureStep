import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    'Optimizer Name': ['SimpleSGDCurvature', 'HeavyBallCurvature', 'NAGCurvature', 'AMSGrad', 'Adam', 'AdamW', 'Adadelta', 'Shampoo', 'NAdam', 'NAdamW', 'Adagrad', 'HeavyBall', 'NAG', 'RMSProp', 'RMSPropMomentum'],
    'Mean Training Loss epoch1': [0.39651, 0.33452, 0.33464, 0.38229, 0.3821, 0.38243, 0.37993, 1.51103, 0.34123, 0.34081, 0.41832, 0.77726, 0.77726, 0.61076, 0.61538],
    'Mean Training Loss epoch2': [0.17593, 0.15764, 0.1574, 0.19564, 0.19528, 0.19519, 0.17809, 0.3873, 0.16887, 0.16921, 0.27517, 0.36868, 0.36867, 0.34963, 0.35682],
    'Mean Training Loss epoch3': [0.13057, 0.12197, 0.12209, 0.14119, 0.1412, 0.1416, 0.13931, 0.28421, 0.1374, 0.1374, 0.24026, 0.32254, 0.32255, 0.33046, 0.33336],
    'Mean Training Loss epoch4': [0.10601, 0.10183, 0.10172, 0.11287, 0.11358, 0.11413, 0.11818, 0.23752, 0.12632, 0.12618, 0.21761, 0.29565, 0.29565, 0.31796, 0.32598],
    'Mean Training Loss epoch5': [0.09036, 0.08857, 0.08846, 0.09475, 0.09664, 0.09725, 0.10502, 0.20485, 0.12249, 0.12287, 0.20109, 0.27446, 0.27446, 0.31396, 0.32099],
    'Mean Training Loss epoch6': [0.07877, 0.0792, 0.07916, 0.08296, 0.08501, 0.08565, 0.09545, 0.18046, 0.12197, 0.12161, 0.18836, 0.25613, 0.25614, 0.30614, 0.32002],
    'Mean Training Loss epoch7': [0.06973, 0.07067, 0.07075, 0.07267, 0.0753, 0.07603, 0.08745, 0.16148, 0.12092, 0.12179, 0.17786, 0.23913, 0.23914, 0.30692, 0.31941],
    'Mean Training Loss epoch8': [0.06195, 0.06429, 0.06407, 0.06531, 0.06788, 0.0693, 0.08107, 0.14628, 0.12473, 0.12504, 0.16904, 0.22388, 0.22388, 0.30691, 0.32144],
    'Mean Training Loss epoch9': [0.05586, 0.05887, 0.05865, 0.05876, 0.06244, 0.06338, 0.0748, 0.13371, 0.13025, 0.12858, 0.16149, 0.21011, 0.21012, 0.30409, 0.32368],
    'Mean Training Loss epoch10': [0.05067, 0.05326, 0.05351, 0.05411, 0.05771, 0.0587, 0.07009, 0.12331, 0.13143, 0.13328, 0.15491, 0.1978, 0.19781, 0.3072, 0.31932]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(14, 8))
for optimizer in df['Optimizer Name']:
    losses = df[df['Optimizer Name'] == optimizer].iloc[0, 1:].values
    plt.plot(range(1, 11), losses, label=optimizer)

plt.xlabel('Epoch')
plt.ylabel('Mean Training Loss')
plt.title('Mean Training Loss over Epochs for Different Optimizers')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('outputs/plots/mnist_optimizer_losses.png')

plt.show()
