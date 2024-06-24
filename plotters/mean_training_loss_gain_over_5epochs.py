import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the given input
data = {
    "Optimizer": ["Adadelta", "Adagrad", "Adam", "AdamW", "AMSGrad", "HeavyBall", "NAdam", "NAdamW", "NAG", "RMSProp", "RMSPropMomentum", "SimpleSGD"],
    "Epoch_1": [0.010991917, 0.024861417, -0.0067625, 0.0108145, 0.002314167, 0.167656917, 0.002429667, 0.020950583, 0.155248917, -0.016485167, 0.002767083, 0.539742504],
    "Epoch_2": [0.071387333, 0.046564833, 0.009065833, 0.023831917, 0.004105417, 0.564890333, -0.003835417, 0.069456333, 0.459523917, 0.046684417, 0.017398167, 1.252146061],
    "Epoch_3": [0.065003, 0.049125417, -0.008171083, 0.010111583, 0.01043375, 0.197533917, 0.001610417, -0.013666561, 0.16732425, 0.015839083, -0.018411167, 0.750951018],
    "Epoch_4": [0.05667075, 0.049988917, 0.008893333, 0.012479333, 0.015027333, 0.280896583, 0.008449667, -0.00336811, 0.237815, 0.0243955, -0.018351833, 0.821385925],
    "Epoch_5": [0.054823333, 0.056792917, 0.003704917, 0.019335167, 0.03454925, 0.379726917, 0.026809167, 0.019472961, 0.322022583, 0.046374417, -0.00997575, 0.908407596]
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(12, 8))
for optimizer in df["Optimizer"]:
    plt.plot(df.columns[1:], df[df["Optimizer"] == optimizer].values[0][1:], marker='o', label=optimizer)

plt.title("Gain in Mean Training Loss by Using Curvature Step Size")
plt.xlabel("Epochs")
plt.ylabel("Mean Training Loss Gain")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Save the plot as an image file
output_file = "outputs/plots/curvature_step_gain_plot.pdf"
plt.savefig(output_file)

# Display the plot
plt.show()

