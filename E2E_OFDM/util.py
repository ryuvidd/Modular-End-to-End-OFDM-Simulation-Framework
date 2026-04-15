import logging
import matplotlib.pyplot as plt

class Plotter():
    def plot_BER(self, SNR, BER, label, save_fig_name):
        plt.figure()

        for i in range(len(label)):
            plt.plot(SNR, BER[i], marker='o', label=label[i])

        plt.yscale('log')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('BER Performance of OFDM System')

        plt.grid(True, which='both')
        plt.legend(label)
        plt.tight_layout()

        plt.savefig(save_fig_name, dpi=300)

    def plot_NMSE(self, SNR, NMSE, label, save_fig_name):
        plt.figure()

        for i in range(len(label)):
            plt.plot(SNR, NMSE[i], marker='o', label=label[i])

        plt.xlabel('SNR (dB)')
        plt.ylabel('NMSE (dB)')
        plt.title('Channel Estimation NMSE')

        plt.grid(True)
        plt.legend(label)
        plt.tight_layout()

        plt.savefig(save_fig_name, dpi=300)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)