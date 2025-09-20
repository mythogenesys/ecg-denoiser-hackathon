# How My AI Works 🤖

Here’s my two-step AI pipeline:

1. **Denoiser**  
   - Architecture: 1D U-Net 🧩  
   - Special Sauce: I added physics-inspired loss terms —  
     *gradient loss* (sharp QRS spikes) + *FFT loss* (right frequencies).  
   - Goal: Turn messy ECG → clean, realistic ECG.

2. **Classifier**  
   - Architecture: Lightweight 1D CNN 📊  
   - Input: Beat-by-beat ECG snippets.  
   - Output: Five standard heartbeat categories (N, S, V, F, Q).  

Here’s the workflow:

![AI Pipeline](img/ai_pipeline.png)

The magic moment?  
When I ran a noisy signal through my denoiser, the classifier’s accuracy jumped from **90% → 96%**.
