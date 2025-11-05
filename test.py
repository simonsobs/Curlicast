import numpy as np
import curlicast as cct
import covariance_validation.noisecalc as nc

config_global = {'nside': 256, 'dell': 10}
config_survey = {'fsky': 0.1}
#config_survey = {'mask': 'covariance_validation/data/mask_apodized.fits',
#                 'nhits': 'covariance_validation/data/nhits_ns256.fits.gz'}
config_sky = {'FGs': {'A_sync': 1.6, 'alpha_sync': -0.93,
                      'A_dust': 28.0, 'alpha_dust': -0.16,
                      'beta_sync': -3.0, 'beta_dust': 1.54,
                      'T_dust': 20.9, 'nu0_sync': 23.0,
                      'nu0_dust': 353.0},
              'cmb': {'Alens': 1.0,
                      'camb_file': 'covariance_validation/data/camb_lens_nobb.dat'}}
config_inst = {'f27': {'bandpass': 27.},
               'f39': {'bandpass': 39.},
               'f93': {'bandpass': 93.},
               'f150': {'bandpass': 150.},
               'f220': {'bandpass': 220.},
               'f280': {'bandpass': 280.}}


class NoiseGenSO(object):
    def __init__(self, fsky=0.1,
                 n_tube_LF=1, n_tube_MF=9, n_tube_UHF=5,
                 one_over_f=0, sensitivity='baseline'):
        ncal = nc.SOSatV3point1(sensitivity_mode=sensitivity,
                                N_tubes=[n_tube_LF, n_tube_MF, n_tube_UHF],
                                one_over_f_mode=one_over_f,
                                survey_years=1.)
        self.fsky = fsky
        self.nc = ncal

    def get_nls(self, lmax):
        ll, _, nl_from2 = self.nc.get_noise_curves(f_sky=self.fsky,
                                                   ell_max=lmax+1,
                                                   delta_ell=1,
                                                   deconv_beam=False)
        nl = np.zeros([len(nl_from2), lmax+1])
        nl[:, 2:] = nl_from2
        return nl

ng = NoiseGenSO(n_tube_LF=1, n_tube_MF=9, n_tube_UHF=5)

print("Generating data")
dg = cct.DataGeneratorPlawFG(config_global=config_global,
                             config_sky=config_sky,
                             config_inst=config_inst,
                             config_survey=config_survey,
                             noise_generator=ng)
s = dg.generate_sacc_file()
s.save_fits("data.fits", overwrite=True)

print("Running compsep")
cs = cct.CompSep('params_test.yml')
cs.run()


print("Analysing chains")
d = np.load("blah/emcee.npz")
nwalkers, nsamples, npar = d['chain'].shape
chain = d['chain'][:, 300:, :].reshape([-1, npar]).T
for n, c in zip(d['names'], chain):
    mean = np.mean(c)
    std = np.std(c)
    print(f'{n} = %.5lf +- %.5lf' % (mean, std))
