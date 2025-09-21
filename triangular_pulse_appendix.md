### B. Detailed Triangular-Pulse Calculation (Worked Example)

This appendix gives a step-by-step derivation comparing an unregularized (L2) reconstruction with a gradient-regularized reconstruction for a discrete triangular pulse corrupted by additive noise. We show analytically how the gradient regularizer preferentially attenuates high-frequency noise and better preserves the pulse's peak slope.

**1. Setup: signal and noise model**

Let the clean signal \(x\in\mathbb{R}^N\) be a discrete triangular pulse centered at \(n=N/2\) with amplitude \(A\) and total width \(w\) (assume \(w\) is even for symmetry). The rising edge gradient is approximately
\[
g_{\text{true}} \approx \frac{A}{w/2}.
\]

The observed signal is
\[
y = x + \eta,
\]
with \(\eta\) additive noise (assumed zero-mean). Denote the reconstruction error \(v=\hat{x}-x\).

**2. Case 1 — Unregularized L2 reconstruction**

Minimizing \(\|y-\hat{x}\|_2^2\) alone gives the trivial solution \(\hat{x}=y\). The forward-difference gradient at a rising-edge index \(i\) is
\[
(G\hat{x})_i = (Gx)_i + (G\eta)_i = g_{\text{true}} + (\eta_{i+1}-\eta_i),
\]
so the peak slope is corrupted by the noise difference \(G\eta\). For high-frequency (rapidly varying) noise, \(|G\eta|\) can be large, producing large peak-slope error.

**3. Case 2 — Gradient-regularized reconstruction**

Add a temporal-gradient penalty and solve
\[
\min_{\hat{x}}\ \|y-\hat{x}\|_2^2 + \alpha \|G\hat{x}\|_2^2,\qquad \alpha>0.
\]
Setting the derivative to zero yields the normal equation
\[
(I + \alpha G^\top G)\,\hat{x} = y.
\]
With \(\Delta := G^\top G\) (the discrete Laplacian), the solution is
\[
\hat{x} = (I + \alpha\Delta)^{-1} y.
\]

**4. Fourier-domain analysis (filter interpretation)**

Under the circulant approximation (i.e., assuming periodic boundary conditions so the DFT diagonalizes \(\Delta\)), \(\Delta\) has eigenvalues
\[
\lambda_k = 2 - 2\cos\!\left(\tfrac{2\pi k}{N}\right) = 4\sin^2\!\left(\tfrac{\pi k}{N}\right),\quad k=0,\dots,N-1.
\]
Hence the reconstruction in the DFT domain is
\[
\widehat{X}[k] = \frac{1}{1+\alpha\lambda_k}\;Y[k],
\]
so the frequency-domain gain is
\[
H(k)=\frac{1}{1+\alpha\lambda_k}.
\]
Key behavior:
- For low frequencies (\(k\) small), \(\lambda_k\approx 0\) so \(H(k)\approx 1\): the filter preserves the pulse's low-frequency shape.
- For high frequencies, \(\lambda_k\) is large and \(H(k)\ll1\): the filter strongly attenuates high-frequency components (i.e., noise).

**5. Error decomposition and gradient attenuation**

Write
\[
\hat{x}=(I+\alpha\Delta)^{-1}(x+\eta),
\]
so the reconstruction error is
\[
v=\hat{x}-x = \big((I+\alpha\Delta)^{-1}-I\big)x + (I+\alpha\Delta)^{-1}\eta.
\]
Thus \(v\) decomposes into a **bias** term \(((I+\alpha\Delta)^{-1}-I)x\), which is a small smoothing of the true pulse (controlled by \(\alpha\)), and a **filtered-noise** term \((I+\alpha\Delta)^{-1}\eta\), which is a low-pass filtered version of the original noise.

Applying the forward difference operator \(G\) gives
\[
Gv = G\big((I+\alpha\Delta)^{-1}-I\big)x + G(I+\alpha\Delta)^{-1}\eta.
\]
Because \(G(I+\alpha\Delta)^{-1}\) has frequency response proportional to \(\tfrac{i\omega_k}{1+\alpha\lambda_k}\), its magnitude is small at high frequencies: the high-frequency components of \(\eta\) are strongly suppressed in \(G(I+\alpha\Delta)^{-1}\eta\). Therefore, the variance of the gradient error is much smaller than that of the unregularized case \(G\eta\), while the bias term only mildly rounds the pulse peak for moderate \(\alpha\).

A compact spectral bound (informal) is
\[
\|G(I+\alpha\Delta)^{-1}\eta\|_2 \le \max_k\Big|\frac{\omega_k}{1+\alpha\lambda_k}\Big|\;\|\widehat{\eta}\|_2,
\]
so increasing \(\alpha\) reduces the multiplicative factor on high-frequency noise components.

**6. Conclusion**

The gradient-regularized solution trades a small bias (slight rounding of sharp corners) for a large reduction in the gradient variance induced by noise. Consequently, the estimated peak slope using the gradient-regularized estimator is closer to the true slope \(g_{\text{true}}\) than the unregularized estimate, demonstrating analytically how the temporal-gradient term in STPC protects morphological features such as sharp onsets.

**Remark.** The circulant approximation simplifies the exposition; for finite-support signals one may adopt Neumann (reflective) or other boundary conditions — the qualitative conclusions about HF attenuation and bias–variance tradeoff remain unchanged.
