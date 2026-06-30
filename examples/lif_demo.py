from neurosim.neurons import LIFNeuron
import matplotlib.pyplot as plt

neuron = LIFNeuron()
result = neuron.simulate(current=2.0, t_max=500, dt=0.1)

plt.figure(figsize=(12, 4))
plt.plot(result["time"], result["voltage"])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Leaky Integrate-and-Fire Neuron Simulation")
plt.tight_layout()
plt.show()

print("Spike count:", result["spike_count"])
print("Firing rate:", result["firing_rate"], "Hz")