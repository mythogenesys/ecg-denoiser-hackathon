# How Can AI Clean an ECG Signal?
Our project uses a type of AI called a **Deep Neural Network**. You can think of it as a very smart filter.

### The "Before and After" Teacher
We taught our AI by showing it thousands of examples of:
1.  A perfectly **clean** ECG signal.
2.  The **same** ECG signal, but with realistic noise added to it.

The AI's job was to learn the mathematical patterns to transform the noisy signal back into the clean one.

### The U-Net: A Special Kind of AI
We used a "U-Net" architecture. It's special because it's great at looking at a signal at different zoom levels:
-   It "zooms out" to see the big picture, like slow baseline wander.
-   It "zooms in" to see the tiny details, like the sharp tip of the R-peak.

By combining both views, it can clean the signal without accidentally blurring out the important diagnostic details.