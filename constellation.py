import numpy as np
import matplotlib.pyplot as plt

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
    # Part 1 et 2 restent identiques
    # [...]

    # Part 3: Constellations BPSK et QPSK avec variation SNR
    n_symbols = 1000
    snr_values = [0, 10, 20]  # Different SNR values in dB

    # Create figure for all constellations
    plt.figure(figsize=(15, 10))
    plt.suptitle('Effet du SNR sur les Constellations BPSK et QPSK', fontsize=14)

    # Generate base symbols
    bpsk_symbols = 2 * (np.random.randint(0, 2, n_symbols) - 0.5)
    qpsk_symbols = (np.random.randint(0, 4, n_symbols) * np.pi/2) + np.pi/4
    qpsk_complex = np.exp(1j * qpsk_symbols)

    # Plot constellations for each SNR
    for i, snr_db in enumerate(snr_values, 1):
        # Calculate noise power based on SNR
        bpsk_signal_power = np.mean(np.abs(bpsk_symbols)**2)
        qpsk_signal_power = np.mean(np.abs(qpsk_complex)**2)
        
        bpsk_noise_power = bpsk_signal_power / (10**(snr_db/10))
        qpsk_noise_power = qpsk_signal_power / (10**(snr_db/10))
        
        # Add noise
        bpsk_noise = np.sqrt(bpsk_noise_power) * np.random.normal(0, 1, n_symbols)
        qpsk_noise = np.sqrt(qpsk_noise_power/2) * (np.random.normal(0, 1, n_symbols) + 
                                                   1j * np.random.normal(0, 1, n_symbols))
        
        bpsk_received = bpsk_symbols + bpsk_noise
        qpsk_received = qpsk_complex + qpsk_noise

        # BPSK Constellation
        plt.subplot(2, 3, i)
        plt.scatter(np.real(bpsk_received), np.imag(bpsk_received), 
                   alpha=0.5, label='Reçu', s=20)
        plt.scatter([-1, 1], [0, 0], color='red', marker='x', 
                   s=100, label='Idéal', linewidth=2)
        plt.title(f'BPSK Constellation (SNR = {snr_db} dB)')
        plt.grid(True)
        plt.axis([-2.5, 2.5, -2.5, 2.5])
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.legend()

        # QPSK Constellation
        plt.subplot(2, 3, i+3)
        plt.scatter(np.real(qpsk_received), np.imag(qpsk_received), 
                   alpha=0.5, label='Reçu', s=20)
        plt.scatter(np.real(qpsk_complex[::50]), np.imag(qpsk_complex[::50]), 
                   color='red', marker='x', s=100, label='Idéal', linewidth=2)
        plt.title(f'QPSK Constellation (SNR = {snr_db} dB)')
        plt.grid(True)
        plt.axis([-2.5, 2.5, -2.5, 2.5])
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.legend()

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save with high DPI for better quality
    plt.savefig('constellations_with_SNR.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Optionnel : Afficher les statistiques d'erreur
    def calculate_error_rate(received, ideal, threshold=0.5):
        if np.iscomplexobj(ideal):
            errors = np.sum(np.abs(received - ideal) > threshold)
        else:
            errors = np.sum(np.abs(received - ideal) > threshold)
        return errors / len(received)

    print("\nTaux d'erreur pour différents SNR :")
    print("SNR (dB) | BPSK Error Rate | QPSK Error Rate")
    print("-" * 45)
    
    for snr_db in snr_values:
        # Recalcule les signaux reçus pour les statistiques
        bpsk_noise_power = 1 / (10**(snr_db/10))
        qpsk_noise_power = 1 / (10**(snr_db/10))
        
        bpsk_received = bpsk_symbols + np.sqrt(bpsk_noise_power) * np.random.normal(0, 1, n_symbols)
        qpsk_received = qpsk_complex + np.sqrt(qpsk_noise_power/2) * (np.random.normal(0, 1, n_symbols) + 
                                                                     1j * np.random.normal(0, 1, n_symbols))
        
        bpsk_error = calculate_error_rate(bpsk_received, bpsk_symbols)
        qpsk_error = calculate_error_rate(qpsk_received, qpsk_complex)
        
        print(f"{snr_db:8d} | {bpsk_error:14.3%} | {qpsk_error:14.3%}")