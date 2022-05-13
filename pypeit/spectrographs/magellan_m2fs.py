"""
Module for M2FS specific methods.
Modified from module for LRIS specific methods.

Modified by Song, Ying-Yi on 06/11/2021

.. include:: ../include/links.rst
"""
import glob
import os

from IPython import embed

from pkg_resources import resource_filename

import numpy as np

from astropy.io import fits
from astropy import time
from astropy.time import Time

from pypeit import msgs
from pypeit import telescopes
from pypeit import io
from pypeit.core import parse
from pypeit.core import framematch
from pypeit.spectrographs import spectrograph
from pypeit.images import detector_container


class MagellanM2FSSpectrograph(spectrograph.Spectrograph):
    """
    Child to handle Magellan/M2FS specific code
    """
    ndet = 1 # was 2 for Keck/LRIS
    telescope = telescopes.MagellanTelescopePar()

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of ``PypeIt`` methods.
        """
        par = super().default_pypeit_par()

        ### added by YYS on 210809
        par['calibrations']['traceframe']['process']['use_overscan'] = False
        par['calibrations']['traceframe']['process']['use_pixelflat'] = False
        par['calibrations']['traceframe']['process']['use_illumflat'] = False

        # Set wave tilts order
        par['calibrations']['slitedges']['edge_thresh'] = 10. # was 15.
        par['calibrations']['slitedges']['max_shift_adj'] = 3.
        par['calibrations']['slitedges']['fit_order'] = 3
        par['calibrations']['slitedges']['sync_center'] = 'nearest'
        par['calibrations']['slitedges']['sync_predict'] = 'nearest'
        #par['calibrations']['slitedges']['auto_pca'] = False
        par['calibrations']['slitedges']['fwhm_gaussian'] = 1.0
        par['calibrations']['slitedges']['fwhm_uniform'] = 1.0
        par['calibrations']['slitedges']['left_right_pca'] = False #True
        par['calibrations']['slitedges']['det_min_spec_length'] = 0.1
        par['calibrations']['slitedges']['fit_min_spec_length'] = 0.1
        par['calibrations']['slitedges']['smash_range'] = [0.2,0.4]
        #par['calibrations']['slitedges']['pca_n'] = 180
        #par['calibrations']['slitedges']['length_range'] = 0.5
        # TODO: I had to increase this from 1. to 2. to deal with
        # Keck_LRIS_red/multi_1200_9000_d680_1x2/ . May need a
        # different solution given that this is binned data and most of
        # the data in the dev suite is unbinned.
        # JXP -- Increased to 6 arcsec.  I don't know how 2 (or 1!) could have worked.
        #par['calibrations']['slitedges']['minimum_slit_length'] = 0.7 # was 6
        #par['calibrations']['slitedges']['minimum_slit_gap'] = 0.5
        #par['calibrations']['slitedges']['gap_offset'] = 0.5
        #par['calibrations']['slitedges']['sobel_mode'] = 'constant'
        #par['calibrations']['slitedges']['follow_span'] = 5
        # 1D wavelengths
        par['calibrations']['wavelengths']['rms_threshold'] = 0.20  # Might be grism dependent
        # Set the default exposure time ranges for the frame typing
        par['calibrations']['biasframe']['exprng'] = [None, 1]
        par['calibrations']['darkframe']['exprng'] = [20, None]     # Modified by YYS; was: [999999, None] means No dark frames
        par['calibrations']['pinholeframe']['exprng'] = [999999, None]  # No pinhole frames
        par['calibrations']['pixelflatframe']['exprng'] = [None, 60]
        par['calibrations']['traceframe']['exprng'] = [None, 60]
        par['calibrations']['standardframe']['exprng'] = [None, 30]

        # Flexure
        # Always correct for spectral flexure, starting with default parameters
        ## par['flexure']['spec_method'] = 'boxcar'
        # Always correct for spatial flexure on science images
        # TODO -- Decide whether to make the following defaults
        #   May not want to do them for LongSlit
        ### par['scienceframe']['process']['spat_flexure_correct'] = True
        ### par['calibrations']['standardframe']['process']['spat_flexure_correct'] = True

        par['scienceframe']['exprng'] = [60, None]
        return par

    def config_specific_par(self, scifile, inp_par=None):
        """
        Modify the ``PypeIt`` parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            scifile (:obj:`str`):
                File to use when determining the configuration and how
                to adjust the input parameters.
            inp_par (:class:`~pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        par = super().config_specific_par(scifile, inp_par=inp_par)

        # Ignore PCA if longslit
        #  This is a little risky as a user could put long into their maskname
        #  But they would then need to over-ride in their PypeIt file
        if scifile is None:
            msgs.error("You have not included a standard or science file in your PypeIt file to determine the configuration")
        if 'long' in self.get_meta_value(scifile, 'decker'):
            par['calibrations']['slitedges']['sync_predict'] = 'nearest'
            # This might only be required for det=2, but we'll see..
            # TODO: Why is this here and not in MagellanM2FSRSpectrograph???
            if self.name == 'keck_lris_red':
                par['calibrations']['slitedges']['edge_thresh'] = 1000.

        return par

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        self.meta = {}
        # Required (core)
        self.meta['ra'] = dict(ext=0, card='RA-D')
        self.meta['dec'] = dict(ext=0, card='DEC-D')
        self.meta['target'] = dict(ext=0, card='OBJECT')
        #TODO: Check decker is correct
        self.meta['decker'] = dict(ext=0, card='SLITNAME')
        self.meta['binning'] = dict(card=None, compound=True)
#        self.meta['binning'] = dict(ext=0, card='BINNING')
        self.meta['mjd'] = dict(ext=0, card=None, compound=True)
        self.meta['exptime'] = dict(ext=0, card='EXPTIME')
        self.meta['airmass'] = dict(ext=0, card='AIRMASS')
        # Extras for config and frametyping
        self.meta['dispname'] = dict(ext=0, card='INSTRUME')
        self.meta['idname'] = dict(ext=0, card='EXPTYPE')
        self.meta['idname'] = dict(ext=0, card='EXPTYPE')

        '''
        # commented out by YYS on 10/09/2021
        self.meta['slide'] = dict(ext=0, card='SLIDE')
        self.meta['lo-elev'] = dict(ext=0, card='LO-ELEV')
        self.meta['hi-elev'] = dict(ext=0, card='HI-ELEV')
        self.meta['hi-azim'] = dict(ext=0, card='HI-AZIM')
        '''

        # Lamps -- Have varied in time..
        #for kk in range(6): # was 12 # This needs to match the length of LAMPS below
        #    self.meta['lampstat{:02d}'.format(kk+1)] = dict(card=None, compound=True)

    def compound_meta(self, headarr, meta_key):
        """
        Methods to generate metadata requiring interpretation of the header
        data, instead of simply reading the value of a header card.

        Args:
            headarr (:obj:`list`):
                List of `astropy.io.fits.Header`_ objects.
            meta_key (:obj:`str`):
                Metadata keyword to construct.

        Returns:
            object: Metadata value read from the header(s).
        """
        if meta_key == 'binning':
            binspatial, binspec = parse.parse_binning(headarr[0]['BINNING'])
            binning = parse.binning2string(binspec, binspatial)
            return binning
        elif meta_key == 'mjd':
            time = '{:s}T{:s}'.format(headarr[0]['UT-DATE'], headarr[0]['UT-TIME'])
            ttime = Time(time, format='isot')
            return ttime.mjd
        elif 'lampstat' in meta_key:
            idx = int(meta_key[-2:])
            curr_date = time.Time(headarr[0]['MJD'], format='mjd') # was MJD-OBS
            # Modern -- Assuming the change occurred with the new red detector
            t_newlamp = time.Time("2014-02-15", format='isot')  # LAMPS changed in Header
            if curr_date > t_newlamp:
                lamp_names = ['FF-THNE', 'FF-THAR', 'FF-NE', 'FF-HGAR', 'FF-XE', 'FF-QRTZ']
                #lamp_names = ['MERCURY', 'NEON', 'ARGON', 'CADMIUM', 'ZINC', 'KRYPTON', 'XENON',
                #              'FEARGON', 'DEUTERI', 'FLAMP1', 'FLAMP2', 'HALOGEN']
                return headarr[0][lamp_names[idx-1]]  # Use this index is offset by 1
            else:  # Original lamps
                plamps = headarr[0]['LAMPS'].split(',')
                # https: // www2.keck.hawaii.edu / inst / lris / instrument_key_list.html
                old_lamp_names = ['MERCURY', 'NEON', 'ARGON', 'CADMIUM', 'ZINC', 'HALOGEN']
                if idx <= 5: # Arcs
                    return ('off' if plamps[idx - 1] == '0' else 'on')
                elif idx == 10:  # Current FLAMP1
                    return headarr[0]['FLIMAGIN'].strip()
                elif idx == 11:  # Current FLAMP2
                    return headarr[0]['FLSPECTR'].strip()
                elif idx == 12:  # Current Halogen slot
                    return ('off' if plamps[len(old_lamp_names)-1] == '0' else 'on')
                else:  # Lamp didn't exist.  Set to None
                    return 'None'
        else:
            msgs.error("Not ready for this compound meta")

    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        return [] #super().configuration_keys() + ['binning']

    def pypeit_file_keys(self):
        """
        Define the list of keys to be output into a standard ``PypeIt`` file.

        Returns:
            :obj:`list`: The list of keywords in the relevant
            :class:`~pypeit.metadata.PypeItMetaData` instance to print to the
            :ref:`pypeit_file`.
        """
        return super().pypeit_file_keys() + ['slide', 'lo-elev', 'hi-elev', 'hi-azim']

    def check_frame_type(self, ftype, fitstbl, exprng=None):
        """
        Check for frames of the provided type.

        Args:
            ftype (:obj:`str`):
                Type of frame to check. Must be a valid frame type; see
                frame-type :ref:`frame_type_defs`.
            fitstbl (`astropy.table.Table`_):
                The table with the metadata for one or more frames to check.
            exprng (:obj:`list`, optional):
                Range in the allowed exposure time for a frame of type
                ``ftype``. See
                :func:`pypeit.core.framematch.check_frame_exptime`.

        Returns:
            `numpy.ndarray`_: Boolean array with the flags selecting the
            exposures in ``fitstbl`` that are ``ftype`` type frames.
        """
        good_exp = framematch.check_frame_exptime(fitstbl['exptime'], exprng)
        #if ftype in ['pinhole', 'dark']:
        if ftype == 'pinhole':
            # Don't type pinhole or dark frames
            return np.zeros(len(fitstbl), dtype=bool)
        elif ftype == 'dark':
            return good_exp & (fitstbl['idname'] == 'Dark')
        elif ftype == 'bias':
            return good_exp & (fitstbl['idname'] == 'Bias')
        elif ftype in ['arc', 'tilt']:
            return fitstbl['idname'] == 'Lamp-ThArNe'
        #elif ftype == 'trace':
        #    return fitstbl['idname'] == 'Lamp-Quartz'
        elif ftype in ['pixelflat', 'trace', 'illumflat']:
            # Flats and trace frames are typed together
            return good_exp & (fitstbl['idname'] == 'Twilight')
        elif ftype == 'science':
            return good_exp & (fitstbl['idname'] == 'Object')
        #elif ftype == 'standard':
        #    return good_exp & self.lamps(fitstbl, 'off') #& (fitstbl['hatch'] == 'open')

        msgs.warn('Cannot determine if frames are of type {0}.'.format(ftype))
        return np.zeros(len(fitstbl), dtype=bool)

    def lamps(self, fitstbl, status):
        """
        Check the lamp status.

        Args:
            fitstbl (`astropy.table.Table`_):
                The table with the fits header meta data.
            status (:obj:`str`):
                The status to check. Can be ``'off'``, ``'arcs'``, or
                ``'dome'``.

        Returns:
            `numpy.ndarray`_: A boolean array selecting fits files that meet
            the selected lamp status.

        Raises:
            ValueError:
                Raised if the status is not one of the valid options.
        """
        if status == 'off':
            # Check if all are off
            return np.all(np.array([ (fitstbl[k] == 'off') | (fitstbl[k] == 'None')
                                        for k in fitstbl.keys() if 'lampstat' in k]), axis=0)
        if status == 'arcs':
            # Check if any arc lamps are on
            arc_lamp_stat = [ 'lampstat{0:02d}'.format(i) for i in range(1,6) ] # was (1,9)
            return np.any(np.array([ fitstbl[k] == 'on' for k in fitstbl.keys()
                                            if k in arc_lamp_stat]), axis=0)
        if status == 'dome':
            # Check if any dome lamps are on
            # Warning 9, 10 are FEARGON and DEUTERI
            dome_lamp_stat = [ 'lampstat{0:02d}'.format(i) for i in range(9,13) ]
            return np.any(np.array([ fitstbl[k] == 'on' for k in fitstbl.keys()
                                            if k in dome_lamp_stat]), axis=0)
        raise ValueError('No implementation for status = {0}'.format(status))

    def subheader_for_spec(self, row_fitstbl, raw_header, extra_header_cards=None,
                           allow_missing=False):
        """
        Generate a dict that will be added to the Header of spectra files
        generated by ``PypeIt`` (e.g. :class:`~pypeit.specobjs.SpecObjs`).

        Args:
            row_fitstbl (dict-like):
                Typically an `astropy.table.Row`_ or
                `astropy.io.fits.Header`_ with keys defined by
                :func:`~pypeit.core.meta.define_core_meta`.
            raw_header (`astropy.io.fits.Header`_):
                Header that defines the instrument and detector, meaning that
                the header must contain the ``INSTRUME``, ``DETECTOR``,
                ``GRANAME``, ``GRISNAME``, and ``SLITNAME`` header cards. If
                provided, this must also contain the header cards provided by
                ``extra_header_cards``.
            extra_header_cards (:obj:`list`, optional):
                Additional header cards from ``raw_header`` to include in the
                output dictionary. Can be an empty list or None.
            allow_missing (:obj:`bool`, optional):
                Ignore any keywords returned by
                :func:`~pypeit.core.meta.define_core_meta` are not present in
                ``row_fitstbl``. Otherwise, raise ``PypeItError``.

        Returns:
            :obj:`dict`: Dictionary with data to include an output fits
            header file or table downstream.
        """
        _extra_header_cards = ['GRANAME', 'GRISNAME', 'SLITNAME']
        if extra_header_cards is not None:
            _extra_header_cards += extra_header_cards
        return super().subheader_for_spec(row_fitstbl, raw_header,
                                          extra_header_cards=_extra_header_cards,
                                          allow_missing=allow_missing)


class MagellanM2FSBSpectrograph(MagellanM2FSSpectrograph):
    """
    Child to handle Magellan/M2FSb specific code
    """

    name = 'magellan_m2fs_blue' # was keck_lris_blue
    camera = 'M2FSb' # was LRISb
    pypeline = 'MultiSlit'
    supported = False # was Ture
    comment = 'Blue camera; see :doc:`lris`'

    def get_detector_par(self, det, hdu):
        """
        Return metadata for the selected detector.

        Args:
            hdu (`astropy.io.fits.HDUList`_):
                The open fits file with the raw image of interest.
            det (:obj:`int`):
                1-indexed detector number.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        # Binning
        # TODO: Could this be detector dependent?
        binning = self.get_meta_value(self.get_headarr(hdu), 'binning')

        # Detector 1
        #detector_dict1 = dict(
        detector_dict = dict(
            binning         = binning,
            det             = 1,
            dataext         = 0,
            specaxis        = 1,
            specflip        = False, # was True for MagE,
            spatflip        = False,
            platescale      = 0.120, #was 0.135,
            darkcurr        = 0.0,
            saturation      = 65535.,
            nonlinear       = 0.99, #was 0.86,
            mincounts       = -1e10,
            numamplifiers   = 1, # was 2,
            gain            = np.atleast_1d(0.68), # was np.atleast_1d([1.55, 1.56]),
            ronoise         = np.atleast_1d(2.4), # was np.atleast_1d([3.9, 4.2]),
            datasec         = np.atleast_1d('[1:4112,1:4096]'),
            #datasec         = np.atleast_1d('[1:4112,1000:3300]'),
            #oscansec        = np.atleast_1d('[1:1024, 0:0]'),
            )
        ## Detector 2
        #detector_dict2 = detector_dict1.copy()
        #detector_dict2.update(dict(
        #    det=2,
        #    dataext=2,
        #    gain=np.atleast_1d([1.63, 1.70]),
        #    ronoise=np.atleast_1d([3.6, 3.6])
        #))

        ## Instantiate
        #detector_dicts = [detector_dict1, detector_dict2]
        #detector = detector_container.DetectorContainer(**detector_dicts[det-1])

        ## Deal with number of amps # YYS: this is important!!!
        #namps = hdu[0].header['NUMAMPS']
        ## The website does not give values for single amp per detector so we take the mean
        ##   of the values provided
        #if namps == 2 or ((namps==4) & (len(hdu)==3)):  # Longslit readout mode is the latter.  This is a hack..
        #    detector.numamplifiers = 1
        #    detector.gain = np.atleast_1d(np.mean(detector.gain))
        #    detector.ronoise = np.atleast_1d(np.mean(detector.ronoise))
        #elif namps == 4:
        #    pass
        #else:
        #    msgs.error("Did not see this namps coming..")

        ## Return
        #return detector
        return detector_container.DetectorContainer(**detector_dict)

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of ``PypeIt`` methods.
        """
        par = super().default_pypeit_par()

        #par['calibrations']['slitedges']['det_min_spec_length'] = 0.1
        #par['calibrations']['slitedges']['fit_min_spec_length'] = 0.2

        # 1D wavelength solution -- Additional parameters are grism dependent
        par['calibrations']['wavelengths']['rms_threshold'] = 2.0  # Might be grism dependent..
        par['calibrations']['wavelengths']['sigdetect'] = 5.0 # was 10.0
        #par['calibrations']['wavelengths']['lamps'] = ['ThAr'] # was ['NeI', 'ArI', 'CdI', 'KrI', 'XeI', 'ZnI', 'HgI']
        #par['calibrations']['wavelengths']['nonlinear_counts'] = self.detector[0]['nonlinear'] * self.detector[0]['saturation']
        #par['calibrations']['wavelengths']['n_first'] = 3
        par['calibrations']['wavelengths']['match_toler'] = 2.5
        #par['calibrations']['wavelengths']['method'] = 'reidentify' # was 'full_template'
        par['calibrations']['wavelengths']['cc_thresh'] = 0.50
        par['calibrations']['wavelengths']['cc_local_thresh'] = 0.50
        par['calibrations']['wavelengths']['fwhm'] = 2.0

        # Allow for longer exposure times on blue side (especially if using the Dome lamps)
        par['calibrations']['pixelflatframe']['exprng'] = [None, 300]
        par['calibrations']['traceframe']['exprng'] = [None, 300]

        return par

    def config_specific_par(self, scifile, inp_par=None):
        """
        Modify the ``PypeIt`` parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            scifile (:obj:`str`):
                File to use when determining the configuration and how
                to adjust the input parameters.
            inp_par (:class:`~pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        # Start with instrument wide
        par = super().config_specific_par(scifile, inp_par=inp_par)

        # Wavelength calibrations
        if self.get_meta_value(scifile, 'dispname') == '300/5000':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'keck_lris_blue_300_d680.fits'
            par['flexure']['spectrum'] = os.path.join(resource_filename('pypeit', 'data/sky_spec/'),
                                                      'sky_LRISb_400.fits')
        elif self.get_meta_value(scifile, 'dispname') == '400/3400':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'keck_lris_blue_400_d560.fits'
            par['flexure']['spectrum'] = os.path.join(resource_filename('pypeit', 'data/sky_spec/'),
                                                  'sky_LRISb_400.fits')
        elif self.get_meta_value(scifile, 'dispname') == '600/4000':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'keck_lris_blue_600_d560.fits'
            par['flexure']['spectrum'] = os.path.join(resource_filename('pypeit', 'data/sky_spec/'),
                                                      'sky_LRISb_600.fits')
        elif self.get_meta_value(scifile, 'dispname') == '1200/3400':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'keck_lris_blue_1200_d460.fits'
            par['flexure']['spectrum'] = os.path.join(resource_filename('pypeit', 'data/sky_spec/'),
                                                      'sky_LRISb_600.fits')

        # FWHM
        binning = parse.parse_binning(self.get_meta_value(scifile, 'binning'))
        par['calibrations']['wavelengths']['fwhm'] = 8.0 / binning[0]

        ### # Slit tracing
        ### # Reduce the slit parameters because the flux does not span the full detector
        ### #   It is primarily on the upper half of the detector (usually)
        ### if self.get_meta_value(scifile, 'dispname') == '300/5000':
        ###     par['calibrations']['slitedges']['smash_range'] = [0.5, 1.]

        # Return
        return par

    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        super().init_meta()
        # Add the name of the dispersing element
        self.meta['dispname'] = dict(ext=0, card='SLIDE') # was GRISNAME

    def bpm(self, filename, det, shape=None, msbias=None):
        """
        Generate a default bad-pixel mask.

        Even though they are both optional, either the precise shape for
        the image (``shape``) or an example file that can be read to get
        the shape (``filename`` using :func:`get_image_shape`) *must* be
        provided.

        Args:
            filename (:obj:`str` or None):
                An example file to use to get the image shape.
            det (:obj:`int`):
                1-indexed detector number to use when getting the image
                shape from the example file.
            shape (tuple, optional):
                Processed image shape
                Required if filename is None
                Ignored if filename is not None
            msbias (`numpy.ndarray`_, optional):
                Master bias frame used to identify bad pixels

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set
            to 1 and an unmasked value set to 0.  All values are set to
            0.
        """
        # Call the base-class method to generate the empty bpm
        bpm_img = super().bpm(filename, det, shape=shape, msbias=msbias)

        # Only defined for det=1
        if det == 1:
            hdu = io.fits_open(filename)
            binspatial, binspec = parse.parse_binning(hdu[0].header['BINNING'])
            hdu.close()
            #msgs.info("Custom bad pixel mask for M2FS with spatial binning=%d\t%d,%d"%(binspatial,np.int(0.1*4096//binspatial),np.int(0.5*4096//binspatial)))

            #bpm_img[:np.int(0.33*4096//binspatial),:] = 1.
            #bpm_img[np.int(0.72*4096//binspatial):,:] = 1.

        return bpm_img


class MagellanM2FSRSpectrograph(MagellanM2FSSpectrograph):
    """
    Child to handle Keck/LRISr specific code
    """
    name = 'magellan_m2fs_red' # was keck_lris_red
    camera = 'M2FSr' # was LRISr
    supported = True
    comment = 'Red camera; see :doc:`lris`'

    def get_detector_par(self, det, hdu):
        """
        Return metadata for the selected detector.

        Args:
            hdu (`astropy.io.fits.HDUList`_):
                The open fits file with the raw image of interest.
            det (:obj:`int`):
                1-indexed detector number.

        Returns:
            :class:`~pypeit.images.detector_container.DetectorContainer`:
            Object with the detector metadata.
        """
        # Binning
        binning = self.get_meta_value(self.get_headarr(hdu), 'binning')  # Could this be detector dependent??

        # Detector 1
        detector_dict1 = dict(
            binning=binning,
            det=1,
            dataext=1,
            specaxis=0,
            specflip=False,
            spatflip=False,
            platescale=0.135,
            darkcurr=0.0,
            saturation=65535.,
            nonlinear=0.76,
            mincounts=-1e10,
            numamplifiers=2,
            gain=np.atleast_1d([1.255, 1.18]),
            ronoise=np.atleast_1d([4.64, 4.76]),
        )
        # Detector 2
        detector_dict2 = detector_dict1.copy()
        detector_dict2.update(dict(
            det=2,
            dataext=2,
            gain=np.atleast_1d([1.191, 1.162]),
            ronoise=np.atleast_1d([4.54, 4.62])
        ))

        # Allow for post COVID detector issues
        t2020_1 = time.Time("2020-06-30", format='isot')  # First run
        t2020_2 = time.Time("2020-07-29", format='isot')  # Second run
        date = time.Time(hdu[0].header['MJD'], format='mjd') # was MJD-OBS

        if date < t2020_1:
            pass
        elif date < t2020_2: # This is for the June 30 2020 run
            msgs.warn("We are using LRISr gain/RN values based on WMKO estimates.")
            detector_dict1['gain'] = np.atleast_1d([37.6])
            detector_dict2['gain'] = np.atleast_1d([1.26])
            detector_dict1['ronoise'] = np.atleast_1d([99.])
            detector_dict2['ronoise'] = np.atleast_1d([5.2])
        else: # This is the 2020 July 29 run
            msgs.warn("We are using LRISr gain/RN values based on WMKO estimates.")
            detector_dict1['gain'] = np.atleast_1d([1.45])
            detector_dict2['gain'] = np.atleast_1d([1.25])
            detector_dict1['ronoise'] = np.atleast_1d([4.47])
            detector_dict2['ronoise'] = np.atleast_1d([4.75])

        # Instantiate
        detector_dicts = [detector_dict1, detector_dict2]
        detector = detector_container.DetectorContainer(**detector_dicts[det-1])

        # Deal with number of amps
        namps = hdu[0].header['NUMAMPS']
        # The website does not give values for single amp per detector so we take the mean
        #   of the values provided
        if namps == 2 or ((namps==4) & (len(hdu)==3)):  # Longslit readout mode is the latter.  This is a hack..
            detector.numamplifiers = 1
            # Long silt mode
            if hdu[0].header['AMPPSIZE'] == '[1:1024,1:4096]':
                idx = 0 if det==1 else 1  # Vid1 for det=1, Vid4 for det=2
                detector.gain = np.atleast_1d(detector.gain[idx])
                detector.ronoise = np.atleast_1d(detector.ronoise[idx])
            else:
                detector.gain = np.atleast_1d(np.mean(detector.gain))
                detector.ronoise = np.atleast_1d(np.mean(detector.ronoise))
        elif namps == 4:
            pass
        else:
            msgs.error("Did not see this namps coming..")

        # Return
        return detector

    @classmethod
    def default_pypeit_par(cls):
        """
        Return the default parameters to use for this instrument.

        Returns:
            :class:`~pypeit.par.pypeitpar.PypeItPar`: Parameters required by
            all of ``PypeIt`` methods.
        """
        par = super().default_pypeit_par()

        par['calibrations']['slitedges']['edge_thresh'] = 30. #20.

        # 1D wavelength solution
        par['calibrations']['wavelengths']['lamps'] = ['NeI', 'ArI', 'CdI', 'KrI', 'XeI', 'ZnI', 'HgI']
        #par['calibrations']['wavelengths']['nonlinear_counts'] = self.detector[0]['nonlinear'] * self.detector[0]['saturation']
        par['calibrations']['wavelengths']['sigdetect'] = 10.0
        # Tilts
        # These are the defaults
        par['calibrations']['tilts']['tracethresh'] = 25
        par['calibrations']['tilts']['spat_order'] = 4
        par['calibrations']['tilts']['spec_order'] = 7
        par['calibrations']['tilts']['maxdev2d'] = 1.0
        par['calibrations']['tilts']['maxdev_tracefit'] = 1.0
        par['calibrations']['tilts']['sigrej2d'] = 5.0

        #  Sky Subtraction
        ### par['reduce']['skysub']['bspline_spacing'] = 0.8

        # Defaults for anything other than 1,1 binning
        #  Rest config_specific_par below if binning is (1,1)
        par['scienceframe']['process']['sigclip'] = 5.
        par['scienceframe']['process']['objlim'] = 5.

        return par

    def config_specific_par(self, scifile, inp_par=None):
        """
        Modify the ``PypeIt`` parameters to hard-wired values used for
        specific instrument configurations.

        Args:
            scifile (:obj:`str`):
                File to use when determining the configuration and how
                to adjust the input parameters.
            inp_par (:class:`~pypeit.par.parset.ParSet`, optional):
                Parameter set used for the full run of PypeIt.  If None,
                use :func:`default_pypeit_par`.

        Returns:
            :class:`~pypeit.par.parset.ParSet`: The PypeIt parameter set
            adjusted for configuration specific parameter values.
        """
        # Start with instrument wide
        par = super().config_specific_par(scifile, inp_par=inp_par)

        # Lacosmic CR settings
        #   Grab the defaults for LRISr
        binning = self.get_meta_value(scifile, 'binning')
        # Unbinned LRISr needs very aggressive LACosmics parameters for 1x1 binning
        if binning == '1,1':
            sigclip = 3.0
            objlim = 3. #0.5
            par['scienceframe']['process']['sigclip'] = sigclip
            par['scienceframe']['process']['objlim'] = objlim

        # Wavelength calibrations
        if self.get_meta_value(scifile, 'dispname') == '400/8500':  # This is basically a reidentify
            par['calibrations']['wavelengths']['reid_arxiv'] = 'keck_lris_red_400.fits'
            par['calibrations']['wavelengths']['method'] = 'full_template'
            par['calibrations']['wavelengths']['sigdetect'] = 20.0
            par['calibrations']['wavelengths']['nsnippet'] = 1
        elif self.get_meta_value(scifile, 'dispname') == '600/5000':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'keck_lris_red_600_5000.fits'
            par['calibrations']['wavelengths']['method'] = 'full_template'
        elif self.get_meta_value(scifile, 'dispname') == '600/7500':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'keck_lris_red_600_7500.fits'
            par['calibrations']['wavelengths']['method'] = 'full_template'
        elif self.get_meta_value(scifile, 'dispname') == '1200/9000':
            par['calibrations']['wavelengths']['reid_arxiv'] = 'keck_lris_red_1200_9000.fits'
            par['calibrations']['wavelengths']['method'] = 'full_template'

        # FWHM
        binning = parse.parse_binning(self.get_meta_value(scifile, 'binning'))
        par['calibrations']['wavelengths']['fwhm'] = 8.0 / binning[0]

        # Return
        return par


    def init_meta(self):
        """
        Define how metadata are derived from the spectrograph files.

        That is, this associates the ``PypeIt``-specific metadata keywords
        with the instrument-specific header cards using :attr:`meta`.
        """
        super().init_meta()
        # Add the name of the dispersing element
        self.meta['dispname'] = dict(ext=0, card='GRANAME')

    def configuration_keys(self):
        """
        Return the metadata keys that define a unique instrument
        configuration.

        This list is used by :class:`~pypeit.metadata.PypeItMetaData` to
        identify the unique configurations among the list of frames read
        for a given reduction.

        Returns:
            :obj:`list`: List of keywords of data pulled from file headers
            and used to constuct the :class:`~pypeit.metadata.PypeItMetaData`
            object.
        """
        return [] #super().configuration_keys() + ['dispangle']

    def bpm(self, filename, det, shape=None, msbias=None):
        """
        Generate a default bad-pixel mask.

        Even though they are both optional, either the precise shape for
        the image (``shape``) or an example file that can be read to get
        the shape (``filename`` using :func:`get_image_shape`) *must* be
        provided.

        Args:
            filename (:obj:`str` or None):
                An example file to use to get the image shape.
            det (:obj:`int`):
                1-indexed detector number to use when getting the image
                shape from the example file.
            shape (tuple, optional):
                Processed image shape
                Required if filename is None
                Ignored if filename is not None
            msbias (`numpy.ndarray`_, optional):
                Master bias frame used to identify bad pixels.

        Returns:
            `numpy.ndarray`_: An integer array with a masked value set
            to 1 and an unmasked value set to 0.  All values are set to
            0.
        """
        # Call the base-class method to generate the empty bpm
        bpm_img = super().bpm(filename, det, shape=shape, msbias=msbias)

        # Only defined for det=2
        if det == 2:
            msgs.info("Using hard-coded BPM for det=2 on LRISr")

            # Get the binning
            hdu = io.fits_open(filename)
            binning = hdu[0].header['BINNING']
            hdu.close()

            # Apply the mask
            xbin = int(binning.split(',')[0])
            badc = 16//xbin
            bpm_img[:,:badc] = 1

            # Mask the end too (this is risky as an edge may appear)
            #  But there is often weird behavior at the ends of these detectors
            bpm_img[:,-10:] = 1

        return bpm_img


def lris_read_amp(inp, ext):
    """
    Read one amplifier of an LRIS multi-extension FITS image

    Args:
        inp (str, astropy.io.fits.HDUList):
            filename or HDUList
        ext (int):
            Extension index

    Returns:
        tuple:
            data
            predata
            postdata
            x1
            y1

    """
    # Parse input
    if isinstance(inp, str):
        hdu = io.fits_open(inp)
    else:
        hdu = inp
    n_ext = len(hdu) - 1  # Number of extensions (usually 4)

    # Get the pre and post pix values
    # for LRIS red POSTLINE = 20, POSTPIX = 80, PRELINE = 0, PRECOL = 12
    head0 = hdu[0].header
    precol = head0['precol']
    postpix = head0['postpix']

    # Deal with binning
    binning = head0['BINNING']
    xbin, ybin = [int(ibin) for ibin in binning.split(',')]
    precol = precol//xbin
    postpix = postpix//xbin

    # get entire extension...
    temp = hdu[ext].data.transpose() # Silly Python nrow,ncol formatting
    tsize = temp.shape
    nxt = tsize[0]

    # parse the DETSEC keyword to determine the size of the array.
    header = hdu[ext].header
    detsec = header['DETSEC']
    x1, x2, y1, y2 = np.array(parse.load_sections(detsec, fmt_iraf=False)).flatten()

    # parse the DATASEC keyword to determine the size of the science region (unbinned)
    datasec = header['DATASEC']
    xdata1, xdata2, ydata1, ydata2 = np.array(parse.load_sections(datasec, fmt_iraf=False)).flatten()

    # grab the components...
    predata = temp[0:precol, :]
    # datasec appears to have the x value for the keywords that are zero
    # based. This is only true in the image header extensions
    # not true in the main header.  They also appear inconsistent between
    # LRISr and LRISb!
    #data     = temp[xdata1-1:xdata2-1,*]
    #data = temp[xdata1:xdata2+1, :]
    if (xdata1-1) != precol:
        msgs.error("Something wrong in LRIS datasec or precol")
    xshape = 1024 // xbin * (4//n_ext)  # Allow for single amp
    if (xshape+precol+postpix) != temp.shape[0]:
        msgs.warn("Unexpected size for LRIS detector.  We expect you did some windowing...")
        xshape = temp.shape[0] - precol - postpix
    data = temp[precol:precol+xshape,:]
    postdata = temp[nxt-postpix:nxt, :]

    # flip in X as needed...
    if x1 > x2:
        xt = x2
        x2 = x1
        x1 = xt
        data = np.flipud(data)

    # flip in Y as needed...
    if y1 > y2:
        yt = y2
        y2 = y1
        y1 = yt
        data = np.fliplr(data)
        predata = np.fliplr(predata)
        postdata = np.fliplr(postdata)

    return data, predata, postdata, x1, y1


def convert_lowredux_pixelflat(infil, outfil):
    """ Convert LowRedux pixelflat to PYPIT format
    Returns
    -------

    """
    # Read
    hdu = io.fits_open(infil)
    data = hdu[0].data

    #
    prihdu = fits.PrimaryHDU()
    hdus = [prihdu]
    prihdu.header['FRAMETYP'] = 'pixelflat'

    # Detector 1
    img1 = data[:,:data.shape[1]//2]
    hdu = fits.ImageHDU(img1)
    hdu.name = 'DET1'
    prihdu.header['EXT0001'] = 'DET1-pixelflat'
    hdus.append(hdu)

    # Detector 2
    img2 = data[:,data.shape[1]//2:]
    hdu = fits.ImageHDU(img2)
    hdu.name = 'DET2'
    prihdu.header['EXT0002'] = 'DET2-pixelflat'
    hdus.append(hdu)

    # Finish
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(outfil, clobber=True)
    print('Wrote {:s}'.format(outfil))


