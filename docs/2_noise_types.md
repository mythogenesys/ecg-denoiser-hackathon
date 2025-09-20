# Types of Noise 🎧

Real-world ECGs aren’t clean — they get messy! Here are the main culprits I worked with:

1. **Muscle noise (EMG)**  
   Looks like fuzzy static from body movement.  
   Example: someone flexing or shivering.

2. **Baseline wander**  
   The whole ECG line drifts up and down.  
   Usually from breathing or loose electrodes.

3. **Electrode motion**  
   Sharp glitches when the contact moves suddenly.  

Here’s what they look like:

![Noisy ECG](img/noise_examples.png)

Why does this matter?  
Because automated systems *hate* noise. My job was to train a denoiser that cleans this up **without erasing the important spikes**.
