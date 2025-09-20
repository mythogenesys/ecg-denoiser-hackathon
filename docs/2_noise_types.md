# Why Do ECGs Get Noisy?
When using portable or low-cost ECG devices, the signal can get messy or "noisy." This makes it hard for doctors (and AI!) to read. Here are the main culprits:

1.  **Muscle Artifact (EMG):** When you move, talk, or even shiver, your muscles create tiny electrical signals. These signals can get picked up by the ECG sensor and make the baseline look fuzzy and chaotic.
2.  **Baseline Wander:** Slow drifting of the entire signal up or down. This is often caused by breathing or slight movements of the sensor on the skin.
3.  **Electrode Motion:** A sudden, sharp spike in the signal caused by the sensor (electrode) briefly losing good contact with the skin. These can sometimes look like a real heartbeat, making them very tricky to deal with.

Our project's AI is specifically trained to recognize and remove these types of noise.