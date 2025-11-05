import numpy as np
import healpy as hp
import sacc
import pymaster as nmt
from .covariance_calculator import (
    CovarianceCalculatorMask, CovarianceCalculatorFsky)


def _fcmb(nu):
    x = 0.017608676067552197*nu
    ex = np.exp(x)
    return ex*(x/(ex-1))**2


def _comp_sed(nu,nu0,beta,temp,typ):
    if typ == 'cmb':
        return _fcmb(nu)
    elif typ == 'dust':
        x_to=0.04799244662211351*nu/temp
        x_from=0.04799244662211351*nu0/temp
        return (nu/nu0)**(1+beta)*(np.exp(x_from)-1)/(np.exp(x_to)-1)*_fcmb(nu0)
    elif typ == 'sync':
        return (nu/nu0)**beta*_fcmb(nu0)
    return None


def _dl_plaw(A,alpha,ls):
    return A*((ls+0.001)/80.)**alpha


def _read_camb(fname, lmax):
    larr_all = np.arange(lmax+1)
    l,dtt,dee,dbb,dte = np.loadtxt(fname,unpack=True)
    l = l.astype(int)
    msk = l <= lmax
    l = l[msk]
    dltt = np.zeros(len(larr_all))
    dltt[l] = dtt[msk]
    dlee = np.zeros(len(larr_all))
    dlee[l] = dee[msk]
    dlbb = np.zeros(len(larr_all))
    dlbb[l] = dbb[msk]
    dlte = np.zeros(len(larr_all))
    dlte[l] = dte[msk]
    return dltt,dlee,dlbb,dlte


class Bpass(object):
    def __init__(self, nu, bnu):
        self.nu = nu
        self.bnu = bnu
        self.dnu = np.zeros_like(self.nu)
        self.dnu[1:] = np.diff(self.nu)
        self.dnu[0] = self.dnu[1]
        # CMB units
        norm = np.sum(self.dnu*self.bnu*self.nu**2*_fcmb(self.nu))
        self.bnu /= norm

    @classmethod
    def from_file(cls, fname):
        nu, bnu = np.loadtxt(fname, unpack=True)
        return cls(nu, bnu)

    @classmethod
    def from_freq(cls, freq):
        nu = np.array([freq-1., freq, freq+1.])
        bnu = np.array([0., 1., 0.])
        return cls(nu, bnu)

    def convolve_sed(self,f):
        sed = np.sum(self.dnu*self.bnu*self.nu**2*f(self.nu))
        return sed


class DataGenerator(object):
    def __init__(self, config_sky, config_survey, noise_generator):
        raise NotImplementedError("General class not callable.")

    def _iterate_cls(self):
        icl = 0
        for i1 in range(self.nfreqs):
            for i2 in range(i1, self.nfreqs):
                yield icl, i1, i2
                icl += 1

    def _get_survey_info_mask(self, config_survey):
        mask = hp.ud_grade(hp.read_map(config_survey['mask']),
                           nside_out=self.nside)
        nhits = hp.ud_grade(hp.read_map(config_survey["nhits"]),
                            nside_out=self.nside)
        f = nmt.NmtField(mask, None, spin=2, purify_b=True, purify_e=True)
        w = nmt.NmtWorkspace.from_fields(f, f, self.bins)
        self.cc = CovarianceCalculatorMask(mask, nhits, self.bins)
        self.bpw = w.get_bandpower_windows()[3, :, 3, :]

    def _get_survey_info_fsky(self, config_survey):
        from scipy.signal import unit_impulse
        fsky = config_survey['fsky']
        self.cc = CovarianceCalculatorFsky(fsky, self.bins)
        self.bpw = self.bins.bin_cell(np.eye(self.lmax+1)).T

    def get_survey_info(self, config_global, config_survey):
        self.nside = config_global['nside']
        self.lmax = 3*self.nside-1
        self.ls = np.arange(self.lmax+1)
        self.bins = nmt.NmtBin.from_nside_linear(self.nside, config_global['dell'])
        self.n_bpw = self.bins.get_n_bands()
        if 'mask' in config_survey:
            self._get_survey_info_mask(config_survey)
        elif 'fsky' in config_survey:
            self._get_survey_info_fsky(config_survey)

    def get_bandpasses(self, config_inst):
        all_bpass = []
        for band in config_inst.values():
            bp = band['bandpass']
            if isinstance(bp, str):
                all_bpass.append(Bpass.from_file(bp))
            elif isinstance(bp, (float, int)):
                all_bpass.append(Bpass.from_freq(bp))
        self.bpss = all_bpass


class DataGeneratorPlawFG(DataGenerator):
    def __init__(self, config_global, config_sky, config_inst, config_survey, noise_generator):
        # Component info
        self.fname_camb = config_sky['cmb']['camb_file']
        self.Alens = config_sky['cmb'].get('Alens', 1.0)
        self.A_sync = config_sky['FGs'].get('A_sync', 1.6)
        self.alpha_sync = config_sky['FGs'].get('alpha_sync', -0.93)
        self.beta_sync = config_sky['FGs'].get('beta_sync', -3.0)
        self.A_dust = config_sky['FGs'].get('A_dust', 28.0)
        self.alpha_dust = config_sky['FGs'].get('alpha_dust', -0.16)
        self.beta_dust = config_sky['FGs'].get('beta_dust', 1.54)
        self.T_dust = config_sky['FGs'].get('T_dust', 20.9)
        self.nu0_sync = config_sky['FGs'].get('nu0_sync', 23.0)
        self.nu0_dust = config_sky['FGs'].get('nu0_dust', 353.0)

        # Sky geometry info
        self.get_survey_info(config_global, config_survey)

        # Instrument info
        self.get_bandpasses(config_inst)
        self.nfreqs = len(self.bpss)

        # Noise generator
        self.noigen = noise_generator

    def get_component_spectra(self):
        dls_sync = _dl_plaw(self.A_sync, self.alpha_sync, self.ls)
        dls_dust = _dl_plaw(self.A_dust, self.alpha_dust, self.ls)
        _, _, dls_cmb, _ = _read_camb(self.fname_camb, self.lmax)
        return dls_sync, dls_dust, self.Alens*dls_cmb

    def get_convolved_seds(self):
        seds = np.zeros([3, self.nfreqs])
        for i, bps in enumerate(self.bpss):
            seds[0, i] = bps.convolve_sed(lambda nu : _comp_sed(nu, None, None, None, 'cmb'))
            seds[1, i] = bps.convolve_sed(lambda nu : _comp_sed(nu, self.nu0_sync, self.beta_sync, None, 'sync'))
            seds[2, i] = bps.convolve_sed(lambda nu : _comp_sed(nu, self.nu0_dust, self.beta_dust, self.T_dust, 'dust'))
        return seds

    def generate_sacc_file(self):
        # Component C_ells (ncomp, nell)
        cl2dl = self.ls * (self.ls+1) / (2*np.pi)
        dl2cl = np.zeros_like(cl2dl)
        dl2cl[self.ls > 0] = 1 / cl2dl[self.ls > 0]
        dl_sync, dl_dust, dl_cmb = self.get_component_spectra()
        cl_comp = np.array([dl_cmb, dl_sync, dl_dust]) * dl2cl[None, :]

        # Component SEDs (ncomp, nfreq)
        seds = self.get_convolved_seds()

        # Frequency C_ells
        cl_freqs = np.zeros([self.nfreqs, self.nfreqs, self.lmax+1])
        for i, sed in enumerate(seds):
            clc = cl_comp[i]
            cl_freqs += sed[:, None, None] * sed[None, :, None] * clc[None, None, :]
        # Convolve with bandpowers
        bpw_freqs = np.sum(cl_freqs[:, :, None, :] * self.bpw[None, None, :, :], axis=-1)

        # Noise power spectra
        nl_freqs = self.noigen.get_nls(self.lmax)

        # Covariance matrix
        ncross = self.nfreqs * (self.nfreqs + 1) // 2
        covar = np.zeros([ncross, self.n_bpw, ncross, self.n_bpw])
        for icli, i1, i2 in self._iterate_cls():
            for iclj, j1, j2 in self._iterate_cls():
                if iclj >= icli:
                    cov_block = self.cc.get_covar(cl_freqs, nl_freqs,
                                                  i1, i2, j1, j2)
                    covar[icli, :, iclj, :] = cov_block
                    if iclj != icli:
                        covar[iclj, :, icli, :] = cov_block.T
                else:
                    continue
        covar = covar.reshape([ncross*self.n_bpw, ncross*self.n_bpw])

        # Sacc file
        s = sacc.Sacc()
        # Tracers
        for i in range(self.nfreqs):
            s.add_tracer("NuMap", 'band%d' % (i+1),
                         quantity='cmb_polarization',
                         spin=2,
                         nu=self.bpss[i].nu,
                         bandpass=self.bpss[i].bnu,
                         ell=self.ls,
                         beam=np.ones_like(self.ls),
                         nu_unit='GHz',
                         map_unit='uK_CMB')
        # Cells
        leff = self.bins.get_effective_ells()
        s_wins = sacc.BandpowerWindow(self.ls, self.bpw.T)
        for icl, i1, i2 in self._iterate_cls():
            n1 = "band%d" % (i1+1)
            n2 = "band%d" % (i2+1)
            s.add_ell_cl('cl_bb', n1, n2, leff, bpw_freqs[i1, i2], window=s_wins)
        # Covariance
        s.add_covariance(covar)

        return s
