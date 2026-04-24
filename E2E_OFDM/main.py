from OFDMSystemBlock import *
from dataclasses import replace

if __name__ == '__main__':

    default_config = OFDMconfig(
        NumSubCarrier = 64,
        TotalNumBlock = 10000,
        NumPilotPerBlock = 32,
        QAMModulation = QAM_MODULATION.QPSK_GRAY,
        LengthCP = 8,
        SNR = np.arange(-10,11,2),
        PilotTypes = PILOT_TYPES.COMB,
        ChannelModel = CHANNEL_MODEL.RAYLEIGH,
        NumTap = 5,
        RegenChannel = 5,
        Estimator = CHANNEL_ESTIMATOR.LS,
        Interpolator = CHANNEL_INTERPOLATOR.LINEAR
    )

    totalconfig = [
        replace(default_config, Estimator=CHANNEL_ESTIMATOR.LS),
        replace(default_config, Estimator=CHANNEL_ESTIMATOR.LMMSE)
    ]
    labels = [totalconfig[k].Estimator.value for k in range(len(totalconfig))]

    BERs = []
    NMSEs = []
    for k in range(len(totalconfig)):
        logging.info(f"===== Initialize OFDM system {k+1} =====")
        system = OFDMSystem(totalconfig[k])
        result = system.run()
        logging.info(f"===== Successfully simulate OFDM system {k+1} =====\n")
        BERs.append(result.BER)
        NMSEs.append(result.ChannelNMSE)

    MergedResults = {
        "BERs": np.stack(BERs, axis=0),
        "ChannelNMSEs": np.stack(NMSEs, axis=0)
    }

    logging.info("Visualizing results\n")
    CurvesPlotter = Plotter()
    CurvesPlotter.plot_BER(
        SNR=default_config.SNR, 
        BER=MergedResults["BERs"], 
        label=labels,
        save_fig_name='results/BER__.png'
    )
    CurvesPlotter.plot_NMSE(
        SNR=default_config.SNR, 
        NMSE=MergedResults["ChannelNMSEs"], 
        label=labels,
        save_fig_name='results/NMSE__.png'
    )            