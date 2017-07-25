import matplotlib
matplotlib.use('Qt5Agg')

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
from ctapipe.image import hillas_parameters, HillasParameterizationError
from ctapipe.io.hessio import hessio_event_source
from sklearn.ensemble import RandomForestRegressor
from ctapipe.reco.hillas_intersection import HillasIntersection
import pickle

class TrainMVA(Tool):
    """

    """
    description = "TrainMVA"
    name = 'ctapipe-TrainMVA'

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

    aliases = Dict(dict(infile='TrainMVA.infile',
                        outfile='TrainMVA.outfile',
                        telescopes='TrainMVA.telescopes',
                        amp_cut='TrainMVA.amp_cut',
                        dist_cut='TrainMVA.dist_cut',
                        tail_cut='TrainMVA.tail_cut',
                        pix_cut='TrainMVA.pix_cut',
                        max_events='TrainMVA.max_events'))


    def setup(self):

        self.geoms = dict()
        self.amp_cut = {"LSTCam": 100,
                        "NectarCam": 100,
                        "FlashCam": 100,
                        "CHEC": 50}

        self.dist_cut = {"LSTCam": 2. * u.deg,
                         "NectarCam": 3.3 * u.deg,
                         "FlashCam": 3. * u.deg,
                         "CHEC": 3.8 * u.deg}

        self.tail_cut = {"LSTCam": (8, 16),
                         "NectarCam": (7, 14),
                         "FlashCam": (7, 14),
                         "CHEC": (3, 6)}

        # Calibrators set to default for now
        self.r1 = HessioR1Calibrator(None, None)
        self.dl0 = CameraDL0Reducer(None, None)
        self.calibrator = CameraDL1Calibrator(None, None)

        # If we don't set this just use everything
        if len(self.telescopes) < 2:
            self.telescopes = None
        self.source = hessio_event_source(self.infile, allowed_tels=self.telescopes)

        self.energy_regressor = {"LSTCam": RandomForestRegressor(),
                                "NectarCam": RandomForestRegressor(),
                                "FlashCam": RandomForestRegressor(),
                                "CHEC": RandomForestRegressor()}

        self.training_variables = {"LSTCam": list(),
                                "NectarCam": list(),
                                "FlashCam": list(),
                                "CHEC": list()}

        self.training_target = {"LSTCam": list(),
                                "NectarCam": list(),
                                "FlashCam": list(),
                                "CHEC": list()}

        self.hillas_reco = HillasIntersection()

    def start(self):

        for event in self.source:
            self.calibrate_event(event)
            self.reconstruct_event(event)

    def finish(self):

        for tel_type in self.energy_regressor:
            if len(self.training_variables[tel_type]) >1:

                self.energy_regressor[tel_type].fit(X=np.asarray(self.training_variables[
                    tel_type]),
                                                    y=np.asarray(self.training_target[
                                                                     tel_type]))
                pkl_file = open(self.outfile+"/"+tel_type+".pkl", 'wb')
                pickle.dump(self.energy_regressor[tel_type], pkl_file)
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

    def preselect(self, hillas, tel_id):
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
                        hillas.width > 0 * u.deg:
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

        tel_x = {}
        tel_y = {}
        tel_type = {}

        hillas_nom = {}

        for tel_id in event.dl0.tels_with_data:
            # Get calibrated image (low gain channel only)
            pmt_signal = event.dl1.tel[tel_id].image[0]

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
                                                              tel_id])

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
                moments = hillas_parameters(nom_coord.x, nom_coord.y, pmt_signal * mask)

            except HillasParameterizationError as e:
                print(e)
                continue

            # Make cut based on Hillas parameters
            if self.preselect(moments, tel_id):
                tel_x[tel_id] = tilt_tel.x
                tel_y[tel_id] = tilt_tel.y
                tel_type[tel_id] = self.geoms[tel_id].cam_id
                hillas_nom[tel_id] = moments

        if len(tel_x)>2:
            fit_result = self.hillas_reco.predict(hillas_nom, tel_x, tel_y,
                                                  array_pointing)

            core_grd = GroundFrame(x=fit_result.core_x, y=fit_result.core_y, z=0*u.m)
            core_tilt = core_grd.transform_to(tilted_system)

            for tel in tel_x:

                impact_dist = np.sqrt(np.power(tel_x[tel]-core_tilt.x,2) +
                                      np.power(tel_y[tel]-core_tilt.y,2))

                train_var = np.array([impact_dist.value, np.log10(hillas_nom[tel].size),
                                      hillas_nom[tel].width.value,
                                      hillas_nom[tel].length.value])

                self.training_variables[tel_type[tel]].append(train_var)
                self.training_target[tel_type[tel]].append(event.mc.energy.value)


def main():
    exe = TrainMVA()
    exe.run()


if __name__ == '__main__':
    main()
