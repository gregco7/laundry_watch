"""Signal -> feature vector.

Turns a raw window of accelerometer samples into the ~15-number fingerprint
the model learns from. Pure numpy, no state.

Pipeline per window: subtract mean -> Hann window -> FFT -> reduce to features.

Functions to implement (fill in one at a time):
- window_signal(samples, n=256, overlap=0.5) -> list of windows
- to_spectrum(window)                         -> magnitude spectrum (mean removed, Hann applied, FFT)
- band_energies(spectrum, fs, bands)          -> 8 log-spaced band energies, 0.5-50 Hz
- spectral_centroid(spectrum, fs)             -> center of mass of the spectrum
- rms(window)                                 -> total energy
- deltas(bands, prev_bands)                   -> change in top-3 bands vs previous window
- rolling_mean(history, band)                 -> short-term context over last 5 windows
- extract(window, prev, history, fs)          -> assembles the full feature vector
"""
