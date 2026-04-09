import logging
import matplotlib.pyplot as plt

class Plotter():
    def plot_BER(self, SNR, BER, save_fig_name):
        plt.figure()

        plt.plot(SNR, BER, marker='o', label='LS')
        # plt.plot(snr_list, ber_lmmse, marker='s', label='LMMSE')

        plt.yscale('log')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('BER Performance of OFDM System')

        plt.grid(True, which='both')
        plt.legend(['LS'])
        plt.tight_layout()

        plt.savefig(save_fig_name, dpi=300)

    def plot_NMSE(self, SNR, NMSE, save_fig_name):
        plt.figure()

        plt.plot(SNR, NMSE, marker='o', label='LS')
        # plt.plot(snr_list, nmse_lmmse, marker='s', label='LMMSE')

        plt.xlabel('SNR (dB)')
        plt.ylabel('NMSE (dB)')
        plt.title('Channel Estimation NMSE')

        plt.grid(True)
        plt.legend(['LS'])
        plt.tight_layout()

        plt.savefig(save_fig_name, dpi=300)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)