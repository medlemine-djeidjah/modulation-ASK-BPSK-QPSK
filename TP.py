import numpy as np
import matplotlib.pyplot as plt

# Part 1
def generate_signal_with_noise(t, snr_db):
    x = 10 * np.sin(2 * np.pi * t)
    P_watts = x ** 2
    x_db = 10 * np.log10(P_watts + 1e-10)
    
    signal_power = np.mean(P_watts)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(t))
    noisy_signal = x + noise
    noisy_power = noisy_signal ** 2
    noisy_db = 10 * np.log10(noisy_power + 1e-10)
    
    return x, P_watts, x_db, noisy_signal, noisy_power, noisy_db




# Part 2
def generate_binary_signal(length, bits):
    return np.repeat(bits, length)

def ask_modulate(t, binary_signal, carrier_freq):
    return binary_signal * np.sin(2 * np.pi * carrier_freq * t)

def psk_modulate(t, binary_signal, carrier_freq):
    phase = np.pi * (binary_signal - 1)  
    return np.sin(2 * np.pi * carrier_freq * t + phase)

def add_awgn(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise




if __name__ == "__main__":
    # Part 1
    t = np.linspace(0, 100, 100)
    snr_db = 20
    x, P_watts, x_db, noisy_signal, noisy_power, noisy_db = generate_signal_with_noise(t, snr_db)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.title('Signal')
    plt.xlabel('Temps (s)')
    plt.ylabel('Volt (V)')
    
    plt.subplot(3, 1, 2)
    plt.plot(t, P_watts)
    plt.title('Puissance du signal')
    plt.xlabel('Temps (s)')
    plt.ylabel('Puissance (W)')
    
    plt.subplot(3, 1, 3)
    plt.plot(t, noisy_signal)
    plt.title('Signal avec bruit')
    plt.xlabel('Temps (s)')
    plt.ylabel('Volt (V)')
    
    plt.tight_layout()
    plt.savefig('part1_plots.png')
    plt.close()

    t = np.linspace(0, 100, 100)
    snr_db = 10
    x, P_watts, x_db, noisy_signal, noisy_power, noisy_db = generate_signal_with_noise(t, snr_db)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.title('Signal')
    plt.xlabel('Temps (s)')
    plt.ylabel('Volt (V)')
    
    plt.subplot(3, 1, 2)
    plt.plot(t, P_watts)
    plt.title('Puissance du signal')
    plt.xlabel('Temps (s)')
    plt.ylabel('Puissance (W)')
    
    plt.subplot(3, 1, 3)
    plt.plot(t, noisy_signal)
    plt.title('Signal avec bruit')
    plt.xlabel('Temps (s)')
    plt.ylabel('Volt (V)')
    
    plt.tight_layout()
    plt.savefig('part1_plots_differentSNR.png')
    plt.close()



    # Part 2
    bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    t = np.linspace(0, 1000, 1000)
    binary_signal = generate_binary_signal(125, bits)
    
    carrier_freq = 0.05
    ask_signal = ask_modulate(t, binary_signal, carrier_freq)
    psk_signal = psk_modulate(t, binary_signal, carrier_freq)
    
    ask_noisy = add_awgn(ask_signal, 10)
    psk_noisy = add_awgn(psk_signal, 10)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(t, binary_signal)
    plt.title('Binary Signal')
    plt.ylim(-2.5, 2.5)
    
    plt.subplot(4, 1, 2)
    plt.plot(t, ask_signal)
    plt.title('ASK modulation')
    plt.ylim(-2.5, 2.5)
    
    plt.subplot(4, 1, 3)
    plt.plot(t, psk_signal)
    plt.title('PSK modulation')
    plt.ylim(-2.5, 2.5)
    
    plt.subplot(4, 1, 4)
    plt.plot(t, psk_noisy)
    plt.title('PSK modulation with AWGN')
    plt.ylim(-2.5, 2.5)
    
    plt.tight_layout()
    plt.savefig('part2_plots.png')
    plt.close()

    # BPSK and QPSK Constellation
    # Generate random symbols
    n_symbols = 1000
    bpsk_symbols = 2 * (np.random.randint(0, 2, n_symbols) - 0.5)
    qpsk_symbols = (np.random.randint(0, 4, n_symbols) * np.pi/2) + np.pi/4
    qpsk_complex = np.exp(1j * qpsk_symbols)
    
    # Add noise
    noise_power = 0.1
    bpsk_received = bpsk_symbols + np.random.normal(0, np.sqrt(noise_power), n_symbols)
    qpsk_received = qpsk_complex + (np.random.normal(0, np.sqrt(noise_power), n_symbols) + 
                                  1j * np.random.normal(0, np.sqrt(noise_power), n_symbols))
    
    # Plot constellations
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(np.real(bpsk_received), np.imag(bpsk_received), alpha=0.5, label='Received')
    plt.scatter([-1, 1], [0, 0], color='red', marker='x', s=100, label='Ideal')
    plt.title('BPSK Constellation')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(np.real(qpsk_received), np.imag(qpsk_received), alpha=0.5, label='Received')
    plt.scatter(np.real(qpsk_complex[::50]), np.imag(qpsk_complex[::50]), 
               color='red', marker='x', s=100, label='Ideal')
    plt.title('QPSK Constellation')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('part3_plots.png')
    plt.close()
