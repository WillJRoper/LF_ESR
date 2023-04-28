import h5py
import numpy as np
import schwimmbad
from functools import partial
from unyt import pc
import pandas as pd


def lum_to_M(Lnu):
    """ Convert L_nu to absolute magnitude (M). If no unit
        provided assumes erg/s/Hz. """

    tenpc = 10*pc
    geo = 4*np.pi*(tenpc.to('cm').value)**2

    return -2.5*np.log10(Lnu/geo)-48.6


def get_lum(ii, tag, bins = np.arange(-26,-16,0.5), inp='FLARES',
            _filter = 'FUV', LF = True, Luminosity='DustModelI',
            data_file="flares.hdf5"):

    if inp == 'FLARES':

        num = str(ii)

        if len(num) == 1:
            num =  '0'+num

        filename = data_file
        num = num+'/'


    else:

        filename = F'./data/EAGLE_{inp}_sp_info.hdf5'
        num = ''

    with h5py.File(filename,'r') as hf:

        lum = np.array(hf[F"{num}/{tag}/Galaxy/BPASS_2.2.1/Chabrier300/"
                          + F"Luminosity/{Luminosity}/{_filter}"])

    if LF == True:

        tmp, edges = np.histogram(lum_to_M(lum), bins = bins)

        return tmp

    else:
        return lum


def get_lum_all(tag, bins = np.arange(-25, -16, 0.5), inp = 'FLARES',
                LF = True, _filter = 'FUV', Luminosity='DustModelI',
                data_file="flares.hdf5"):

    if inp == 'FLARES':

        df = pd.read_csv('weights_grid.txt')
        weights = np.array(df['weights'])
        sims = np.arange(0,len(weights))

        calc = partial(get_lum, tag = tag, bins = bins, inp = inp, LF = LF,
                       _filter = _filter, Luminosity = Luminosity,
                       data_file=data_file)

        pool = schwimmbad.MultiPool(processes=12)
        dat = np.array(list(pool.map(calc, sims)))
        pool.close()

        if LF:
            hist = np.sum(dat, axis = 0)
            out = np.zeros(len(bins)-1)
            err = np.zeros(len(bins)-1)
            out_up = np.zeros(len(bins)-1)
            out_low = np.zeros(len(bins)-1)
            for ii, sim in enumerate(sims):
                out+=dat[ii]*weights[ii]
                err+=np.square(np.sqrt(dat[ii])*weights[ii])

            return out, hist, np.sqrt(err)

        else:
            return dat

    else:

        out = get_lum(00, tag = tag, bins = bins, inp = inp, _filter = _filter,
                      Luminosity = Luminosity)

        return out
