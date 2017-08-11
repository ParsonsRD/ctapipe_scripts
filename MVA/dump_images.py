import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy

import astropy.units as u
from traitlets import Dict, List, Unicode, Int, Bool
import numpy as np

from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import HessioR1Calibrator

from ctapipe.coordinates import *
from ctapipe.core import Tool
from ctapipe.image import tailcuts_clean, dilate
from ctapipe.instrument import CameraGeometry
from ctapipe.reco.ImPACT import ImPACTReconstructor
from ctapipe.image import hillas_parameters, HillasParameterizationError
from ctapipe.io.hessio import hessio_event_source

from ctapipe.reco.hillas_intersection import HillasIntersection
from ctapipe.image import FullIntegrator
import ctapipe.visualization as visualization
from ctapipe.io.containers import (ReconstructedShowerContainer,
                                   ReconstructedEnergyContainer)
from scipy.interpolate import LinearNDInterpolator, Rbf

from ctapipe.plotting.event_viewer import EventViewer
from astropy.io import fits


class ImPACTReconstruction(Tool):
    """

    """
    description = "ImPACTReco"
    name = 'ctapipe-ImPACT-reco'

    infile = Unicode(help='input simtelarray file').tag(config=True)

    outfile = Unicode(help='output fits table').tag(config=True)

    telescopes = List(Int, None, allow_none=True,
                      help='Telescopes to include from the event file. '
                           'Default = All telescopes').tag(config=True)

    max_events = Int(default_value=1000000000,
                     help="Max number of events to include in analysis").tag(config=True)

    amp_cut = Dict().tag(config=True)
    dist_cut = Dict().tag(config=True)
    tail_cut = Dict().tag(config=True)
    pix_cut = Dict().tag(config=True)

    minimiser = Unicode(default_value="minuit",
                        help='minimiser used for ImPACTReconstruction').tag(config=True)

    aliases = Dict(dict(infile='ImPACTReconstruction.infile',
                        outfile='ImPACTReconstruction.outfile',
                        telescopes='ImPACTReconstruction.telescopes',
                        amp_cut='ImPACTReconstruction.amp_cut',
                        dist_cut='ImPACTReconstruction.dist_cut',
                        tail_cut='ImPACTReconstruction.tail_cut',
                        pix_cut='ImPACTReconstruction.pix_cut',
                        minimiser='ImPACTReconstruction.minimiser',
                        max_events='ImPACTReconstruction.max_events'))

    def setup(self):

        self.geoms = dict()
        if len(self.amp_cut) == 0:
            self.amp_cut = {"LSTCam": 92.7,
                            "NectarCam": 90.6,
                            "FlashCam": 90.6,
                            "GATE": 29.3}
        if len(self.dist_cut) == 0:
            self.dist_cut = {"LSTCam": 1.74 * u.deg,
                             "NectarCam": 3. * u.deg,
                             "FlashCam": 3. * u.deg,
                             "GATE": 3.55 * u.deg}
        if len(self.tail_cut) == 0:
            self.tail_cut = {"LSTCam": (8, 16),
                             "NectarCam": (7, 14),
                             "FlashCam": (7, 14),
                             "GATE": (3, 6)}
        if len(self.pix_cut) == 0:
            self.pix_cut = {"LSTCam": 5,
                            "NectarCam": 4,
                            "FlashCam": 4,
                            "GATE": 4}

        # Calibrators set to default for now
        self.r1 = HessioR1Calibrator(None, None)
        self.dl0 = CameraDL0Reducer(None, None)
        self.calibrator = CameraDL1Calibrator(None, None,
                                              extractor=FullIntegrator(None, None))

        # If we don't set this just use everything
        if len(self.telescopes) < 2:
            self.telescopes = None

        self.source = hessio_event_source(self.infile, allowed_tels=self.telescopes,
                                          max_events=self.max_events)

        self.fit = HillasIntersection()

        self.viewer = EventViewer(draw_hillas_planes=True)

        self.output = fits.HDUList()


    def start(self):

        for event in self.source:
            self.calibrate_event(event)
            self.reconstruct_event(event)

    def finish(self):
        self.output.writeto(self.outfile)

        return True

    def calibrate_event(self, event):
        """
        Run standard calibrators to get from r0 to dl1

        Parameters
        ----------
        event: ctapipe event container

        Returns
        -------
            None
        """
        self.r1.calibrate(event)
        self.dl0.reduce(event)
        self.calibrator.calibrate(event)  # calibrate the events

    def preselect(self, hillas, npix, tel_id):
        """
        Perform pre-selection of telescopes (before reconstruction) based on Hillas
        Parameters

        Parameters
        ----------
        hillas: ctapipe Hillas parameter object
        tel_id: int
            Telescope ID number

        Returns
        -------
            bool: Indicate whether telescope passes cuts
        """
        if hillas is None:
            return False

        # Calculate distance of image centroid from camera centre
        dist = np.sqrt(hillas.cen_x * hillas.cen_x + hillas.cen_y * hillas.cen_y)

        # Cut based on Hillas amplitude and nominal distance
        if hillas.size > self.amp_cut[self.geoms[tel_id].cam_id] and dist < \
                self.dist_cut[self.geoms[tel_id].cam_id] and \
                        hillas.width > 0 * u.deg and \
                        npix > self.pix_cut[self.geoms[tel_id].cam_id]:
            return True

        return False

    def reconstruct_event(self, event):
        """
        Perform full event reconstruction, including Hillas and ImPACT analysis.

        Parameters
        ----------
        event: ctapipe event container

        Returns
        -------
            None
        """
        # store MC pointing direction for the array
        array_pointing = HorizonFrame(alt=event.mcheader.run_array_direction[1] * u.rad,
                                      az=event.mcheader.run_array_direction[0] * u.rad)
        tilted_system = TiltedGroundFrame(pointing_direction=array_pointing)

        image = {}
        pixel_x = {}
        pixel_y = {}
        pixel_area = {}
        tel_type = {}
        tel_x = {}
        tel_y = {}

        hillas = {}
        hillas_nom = {}
        image_pred = {}
        mask_dict = {}
        print("Event energy", event.mc.energy)

        for tel_id in event.dl0.tels_with_data:

            # Get calibrated image (low gain channel only)
            pmt_signal = event.dl1.tel[tel_id].image[0]
            if len(event.dl1.tel[tel_id].image) > 1:
                print(event.dl1.tel[tel_id].image[1][pmt_signal > 100])
                pmt_signal[pmt_signal > 100] = \
                    event.dl1.tel[tel_id].image[1][pmt_signal > 100]
            # Create nominal system for the telescope (this should later used telescope
            # pointing)
            nom_system = NominalFrame(array_direction=array_pointing,
                                      pointing_direction=array_pointing)

            # Create camera system of all pixels
            pix_x, pix_y = event.inst.pixel_pos[tel_id]
            fl = event.inst.optical_foclen[tel_id]
            if tel_id not in self.geoms:
                self.geoms[tel_id] = CameraGeometry.guess(pix_x, pix_y,
                                                          event.inst.optical_foclen[
                                                              tel_id],
                                                          apply_derotation=False)
            # Transform the pixels positions into nominal coordinates
            camera_coord = CameraFrame(x=pix_x, y=pix_y, z=np.zeros(pix_x.shape) * u.m,
                                       focal_length=fl,
                                       rotation=-1 * self.geoms[tel_id].cam_rotation)

            nom_coord = camera_coord.transform_to(nom_system)
            tx, ty, tz = event.inst.tel_pos[tel_id]

            # ImPACT reconstruction is performed in the tilted system,
            # so we need to transform tel positions
            grd_tel = GroundFrame(x=tx, y=ty, z=tz)
            tilt_tel = grd_tel.transform_to(tilted_system)

            # Clean image using split level cleaning
            mask = tailcuts_clean(self.geoms[tel_id], pmt_signal,
                                  picture_thresh=self.tail_cut[self.geoms[
                                      tel_id].cam_id][1],
                                  boundary_thresh=self.tail_cut[self.geoms[
                                      tel_id].cam_id][0])

            # Perform Hillas parameterisation
            moments = None
            try:
                moments_cam = hillas_parameters(event.inst.pixel_pos[tel_id][0],
                                                event.inst.pixel_pos[tel_id][1],
                                                pmt_signal * mask)

                moments = hillas_parameters(nom_coord.x, nom_coord.y, pmt_signal * mask)

            except HillasParameterizationError as e:
                print(e)
                continue

            # Make cut based on Hillas parameters
            if self.preselect(moments, np.sum(mask), tel_id):

                # Dialte around edges of image
                for i in range(4):
                    mask = dilate(self.geoms[tel_id], mask)

                # Save everything in dicts for reconstruction later
                pixel_area[tel_id] = self.geoms[tel_id].pix_area / (fl * fl)
                pixel_area[tel_id] *= u.rad * u.rad
                pixel_x[tel_id] = nom_coord.x
                pixel_y[tel_id] = nom_coord.y

                tel_x[tel_id] = tilt_tel.x
                tel_y[tel_id] = tilt_tel.y

                tel_type[tel_id] = self.geoms[tel_id].cam_id
                image[tel_id] = pmt_signal
                image_pred[tel_id] = np.zeros(pmt_signal.shape)

                hillas[tel_id] = moments_cam
                hillas_nom[tel_id] = moments
                mask_dict[tel_id] = mask

        # Cut on number of telescopes remaining
        if len(image) > 1:
            fit_result = self.fit.predict(hillas_nom, tel_x, tel_y, array_pointing)

            for tel_id in hillas_nom:

                core_grd = GroundFrame(x=fit_result.core_x, y=fit_result.core_y, z=0.*u.m)
                core_tilt = core_grd.transform_to(tilted_system)
                impact = np.sqrt(np.power(tel_x[tel_id]-core_tilt.x,2) +
                                 np.power(tel_y[tel_id]-core_tilt.y,2) )

                pix = np.array([pixel_x[tel_id].to(u.deg).value, pixel_y[tel_id].to(u.deg).value])
                img = image[tel_id]
                #img[img < 0.0] = 0.0
                #img /= np.max(img)

                lin = LinearNDInterpolator(pix.T, img, fill_value=0.0)
                x_img = np.arange(-4, 4, 0.05) * u.deg
                y_img = np.arange(-4, 4, 0.05) * u.deg
                xv, yv = np.meshgrid(x_img, y_img)
                pos = np.array([xv.ravel(), yv.ravel()])
                npix = xv.shape[0]
                img_int = np.reshape(lin(pos.T), xv.shape)
                print(img_int)

                hdu = fits.ImageHDU(img_int.astype(np.float32))
                hdu.header["IMPACT"] = impact.to(u.m).value
                hdu.header["TELTYPE"] = tel_type[tel_id]
                hdu.header["TELNUM"] = tel_id

                hdu.header["MCENERGY"] = event.mc.energy.to(u.TeV).value
                hdu.header["RECENERGY"] = 0
                hdu.header["AMP"] = hillas_nom[tel_id].size
                hdu.header["WIDTH"] = hillas_nom[tel_id].width.to(u.deg).value
                hdu.header["LENGTH"] = hillas_nom[tel_id].length.to(u.deg).value
                self.output.append(hdu)

def main():
    exe = ImPACTReconstruction()
    exe.run()


if __name__ == '__main__':
    main()