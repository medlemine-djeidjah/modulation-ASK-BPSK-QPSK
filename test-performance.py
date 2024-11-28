import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import time

class ModulationPerformance:
    def __init__(self, n_bits=100000):
        self.n_bits = n_bits
        
    def generate_bits(self):
        """Génère des bits aléatoires"""
        return np.random.randint(0, 2, self.n_bits)
    
    def bpsk_modulate(self, bits):
        """Module les bits en BPSK"""
        return 2 * bits - 1
    
    def qpsk_modulate(self, bits):
        """Module les bits en QPSK"""
        # Groupe les bits par paires
        bits_reshaped = bits[:2*(len(bits)//2)].reshape(-1, 2)
        # Convertit chaque paire en symbole QPSK
        symbols = np.zeros(len(bits_reshaped), dtype=complex)
        
        # Mapping Gray code
        for i, bit_pair in enumerate(bits_reshaped):
            if bit_pair[0] == 0 and bit_pair[1] == 0:
                symbols[i] = (1 + 1j) / np.sqrt(2)  # 45 degrés
            elif bit_pair[0] == 0 and bit_pair[1] == 1:
                symbols[i] = (-1 + 1j) / np.sqrt(2)  # 135 degrés
            elif bit_pair[0] == 1 and bit_pair[1] == 1:
                symbols[i] = (-1 - 1j) / np.sqrt(2)  # 225 degrés
            else:
                symbols[i] = (1 - 1j) / np.sqrt(2)  # 315 degrés
                
        return symbols
    
    def add_noise(self, signal, snr_db):
        """Ajoute du bruit blanc gaussien au signal"""
        # Calcul de la puissance du bruit
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(snr_db/10))
        
        if np.iscomplexobj(signal):
            # Pour les signaux complexes (QPSK)
            noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, len(signal)) + 
                                            1j * np.random.normal(0, 1, len(signal)))
        else:
            # Pour les signaux réels (BPSK)
            noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(signal))
            
        return signal + noise
    
    def bpsk_demodulate(self, received):
        """Démodule le signal BPSK"""
        return (received > 0).astype(int)
    
    def qpsk_demodulate(self, received):
        """Démodule le signal QPSK"""
        # Décision basée sur les quadrants
        bits = np.zeros(2*len(received), dtype=int)
        
        for i, symbol in enumerate(received):
            if symbol.real > 0 and symbol.imag > 0:
                bits[2*i:2*i+2] = [0, 0]
            elif symbol.real < 0 and symbol.imag > 0:
                bits[2*i:2*i+2] = [0, 1]
            elif symbol.real < 0 and symbol.imag < 0:
                bits[2*i:2*i+2] = [1, 1]
            else:
                bits[2*i:2*i+2] = [1, 0]
                
        return bits
    
    def calculate_ber(self, original, received):
        """Calcule le taux d'erreur binaire"""
        return np.sum(original != received) / len(original)
    
    def theoretical_ber_bpsk(self, snr_db):
        """Calcule le BER théorique pour BPSK"""
        snr = 10**(snr_db/10)
        return 0.5 * erfc(np.sqrt(snr))
    
    def theoretical_ber_qpsk(self, snr_db):
        """Calcule le BER théorique pour QPSK"""
        snr = 10**(snr_db/10)
        return self.theoretical_ber_bpsk(snr_db)  # Même que BPSK pour QPSK avec Gray coding
    
    def simulate(self, snr_range):
        """Simule les transmissions BPSK et QPSK pour différents SNR"""
        ber_bpsk = []
        ber_qpsk = []
        theo_ber_bpsk = []
        theo_ber_qpsk = []
        
        for snr in snr_range:
            # BPSK
            bits = self.generate_bits()
            bpsk_signal = self.bpsk_modulate(bits)
            received_bpsk = self.add_noise(bpsk_signal, snr)
            demod_bpsk = self.bpsk_demodulate(received_bpsk)
            ber_bpsk.append(self.calculate_ber(bits, demod_bpsk))
            theo_ber_bpsk.append(self.theoretical_ber_bpsk(snr))
            
            # QPSK
            qpsk_signal = self.qpsk_modulate(bits)
            received_qpsk = self.add_noise(qpsk_signal, snr)
            demod_qpsk = self.qpsk_demodulate(received_qpsk)
            ber_qpsk.append(self.calculate_ber(bits[:2*(len(bits)//2)], 
                                             demod_qpsk[:2*(len(bits)//2)]))
            theo_ber_qpsk.append(self.theoretical_ber_qpsk(snr))
            
        return np.array(ber_bpsk), np.array(ber_qpsk), \
               np.array(theo_ber_bpsk), np.array(theo_ber_qpsk)

    def plot_constellation(self, snr_db):
        """Affiche les constellations pour un SNR donné"""
        bits = self.generate_bits()
        
        # BPSK
        bpsk_signal = self.bpsk_modulate(bits)
        received_bpsk = self.add_noise(bpsk_signal, snr_db)
        
        # QPSK
        qpsk_signal = self.qpsk_modulate(bits)
        received_qpsk = self.add_noise(qpsk_signal, snr_db)
        
        plt.figure(figsize=(12, 5))
        
        # BPSK Constellation
        plt.subplot(121)
        plt.scatter(np.real(received_bpsk), np.imag(received_bpsk), 
                   alpha=0.5, label='Reçu')
        plt.scatter([-1, 1], [0, 0], color='red', marker='x', 
                   s=100, label='Idéal')
        plt.title(f'BPSK Constellation (SNR = {snr_db} dB)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        # QPSK Constellation
        plt.subplot(122)
        plt.scatter(np.real(received_qpsk), np.imag(received_qpsk), 
                   alpha=0.5, label='Reçu')
        ideal_qpsk = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        plt.scatter(np.real(ideal_qpsk), np.imag(ideal_qpsk), 
                   color='red', marker='x', s=100, label='Idéal')
        plt.title(f'QPSK Constellation (SNR = {snr_db} dB)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        plt.tight_layout()
        return plt.gcf()

def main():
    # Paramètres de simulation
    n_bits = 100000
    snr_range = np.arange(0, 21, 2)  # 0 à 20 dB
    
    # Création de l'objet de simulation
    mod_perf = ModulationPerformance(n_bits)
    
    # Mesure du temps d'exécution
    start_time = time.time()
    
    # Simulation
    print("Simulation en cours...")
    ber_bpsk, ber_qpsk, theo_ber_bpsk, theo_ber_qpsk = mod_perf.simulate(snr_range)
    
    print(f"Temps d'exécution: {time.time() - start_time:.2f} secondes")
    
    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_bpsk, 'bo-', label='BPSK Simulé')
    plt.semilogy(snr_range, ber_qpsk, 'rs-', label='QPSK Simulé')
    plt.semilogy(snr_range, theo_ber_bpsk, 'b--', label='BPSK Théorique')
    plt.semilogy(snr_range, theo_ber_qpsk, 'r--', label='QPSK Théorique')
    
    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Taux d\'Erreur Binaire (BER)')
    plt.title('Comparaison des Performances BPSK vs QPSK')
    plt.legend()
    plt.savefig('ber_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Afficher les constellations pour quelques valeurs de SNR
    snr_values = [5, 10, 15]
    for snr in snr_values:
        fig = mod_perf.plot_constellation(snr)
        fig.savefig(f'constellation_snr_{snr}dB.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Afficher les résultats numériques
    print("\nRésultats de la simulation:")
    print("SNR (dB) | BPSK BER  | QPSK BER")
    print("-" * 35)
    for i, snr in enumerate(snr_range):
        print(f"{snr:8.1f} | {ber_bpsk[i]:.2e} | {ber_qpsk[i]:.2e}")

if __name__ == "__main__":
    main()